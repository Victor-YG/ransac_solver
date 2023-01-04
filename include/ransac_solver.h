#pragma once

#include "stdlib.h"

#include <math.h>
#include <vector>
#include <iostream>


namespace RANSAC {

// Option
struct Option
{
    unsigned int max_iterations = 100;
    bool final_model_fitting = false;
    bool early_termination = false;
    float concensus_ratio = 0.8;
};


// Model
template <class Element, class ModelParams>
class Model
{
public:
    virtual unsigned int NumElementsRequired() = 0;

    virtual void SetModelParams(const ModelParams& params) = 0;

    virtual ModelParams Fit(
        const std::vector<Element>& elements,
        const std::vector<float>& weights) = 0;

    virtual bool IsInlier(const Element& element, float& loss) = 0;
};


// Solver
template <class Element, class ModelParams>
class Solver
{
public:
    Solver(Model<Element, ModelParams>* model): m_model(model) {};

    void SetOptions(const Option& options)
    {
        m_options = options;
    }

    ModelParams Solve(
        const std::vector<Element>& elements,
        const std::vector<float>& weights,
        std::vector<bool>& labels,
        std::vector<float>& losses)
    {
        labels.clear();
        losses.clear();

        unsigned int size = elements.size();
        unsigned int concensus_count_threshold = ceil(size * m_options.concensus_ratio);

        labels.reserve(size);
        losses.reserve(size);

        unsigned int num = m_model->NumElementsRequired();
        std::vector<Element> elements_selected;
        std::vector<float> weights_selected;
        std::vector<bool> is_inlier_temp;
        ModelParams params_temp;
        ModelParams params_final;
        is_inlier_temp.resize(size);

        int best_count = -1;
        float best_score = std::numeric_limits<float>::lowest();
        std::vector<bool> is_inlier_final;
        is_inlier_final.resize(size);

        for (int i = 0; i < m_options.max_iterations; i++)
        {
            elements_selected.clear();
            weights_selected.clear();

            // random sampling
            for (int n = 0; n < num; n++)
            {
                int idx = rand() % size;
                elements_selected.emplace_back(elements[idx]);
                weights_selected.emplace_back(weights[i]);
            }

            // fit model params
            params_temp = m_model->Fit(elements_selected, weights_selected);

            // count inliers (vs. outliers)
            int count = 0;
            float total_loss = 0;
            for (int i = 0; i < size; i++)
            {
                float loss = 0;
                if (m_model->IsInlier(elements[i], loss))
                {
                    is_inlier_temp[i] = true;
                    total_loss += loss;
                    count++;
                }
                else
                {
                    is_inlier_temp[i] = false;
                }
            }

            // update best belief
            if (count > best_count)
            {
                best_count = count;
                params_final = params_temp;

                for (int i = 0; i < size; i++)
                {
                    is_inlier_final[i] = is_inlier_temp[i];
                }
            }

            // early termination when concensus found
            if (m_options.early_termination &&
                count > concensus_count_threshold) break;
        }

        // compute final params with inliers
        std::vector<Element> inliers;
        std::vector<float> inlier_weights;
        for (int i = 0; i < size; i++)
        {
            if (is_inlier_final[i] == true)
            {
                inliers.emplace_back(elements[i]);
                inlier_weights.emplace_back(weights[i]);
            }
        }

        // final fitting
        if (m_options.final_model_fitting)
        {
            if (inliers.size() > num)
            {
                params_final = m_model->Fit(inliers, inlier_weights);
            }
            else // failed to reject outliers; fit with all elements
            {
                params_final = m_model->Fit(elements, weights);
            }
        }

        // update inlier info and losses
        m_model->SetModelParams(params_final);
        for (int i = 0; i < size; i++)
        {
            float loss = 0.0;
            bool is_inlier = m_model->IsInlier(elements[i], loss);
            labels.emplace_back(is_inlier);
            losses.emplace_back(loss);
        }

        return params_final;
    };

private:
    Model<Element, ModelParams>*    m_model;
    Option                          m_options;
};

} // RANSAC
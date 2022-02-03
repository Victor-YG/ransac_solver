#pragma once

#include "stdlib.h"

#include <vector>
#include <iostream>


namespace RANSAC {

// Model
template <class Element, class ModelParams>
class Model
{
public:
    virtual unsigned int NumElementsRequired() = 0;

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
    Solver(Model<Element, ModelParams>* model_, int max_iteration_)
        : m_model(model_), m_max_iteration(max_iteration_) {};

    ModelParams Solve(const std::vector<Element>& elements)
    {
        unsigned int N = elements.size();
        std::vector<float> weights(N, 1.0);
        std::vector<bool> labels(N, true);
        std::vector<float> losses(N, 0.0);

        return this->Solve(elements, weights, labels, losses);
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
        labels.reserve(size);
        losses.reserve(size);

        unsigned int num = m_model->NumElementsRequired();
        std::vector<Element> elements_selected;
        std::vector<float> weights_selected;
        std::vector<bool> is_inlier_temp;
        is_inlier_temp.resize(size);

        int best_count = -1;
        float best_score = std::numeric_limits<float>::lowest();
        std::vector<bool> is_inlier_final;
        is_inlier_final.resize(size);

        for (int i  = 0; i < m_max_iteration; i++)
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
            m_model->Fit(elements_selected, weights_selected);
            
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
                for (int i = 0; i < size; i++)
                {
                    is_inlier_final[i] = is_inlier_temp[i];
                }
            }

            // early termination when concensus found 
            // TODO::make this as an option
            if (count > size * 0.8) break;
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
        ModelParams final_params;
        if (inliers.size() > num)
        {
            final_params = m_model->Fit(inliers, inlier_weights);
        }
        else // failed to reject outliers
        {
            final_params = m_model->Fit(elements, weights);
        }

        // update inlier info and losses
        for (int i = 0; i < size; i++)
        {
            float loss = 0.0;
            bool is_inlier = m_model->IsInlier(elements[i], loss);
            labels.emplace_back(is_inlier);
            losses.emplace_back(loss);
        }

        return final_params;
    };

private:
    int                             m_max_iteration;
    Model<Element, ModelParams>*    m_model;
};

} // RANSAC
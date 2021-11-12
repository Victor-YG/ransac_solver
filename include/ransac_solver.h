#pragma once

#include <vector>
#include <iostream>
#include "stdlib.h"


namespace RANSAC {

// Model
template <class Element, class ModelParams>
class Model
{
public:
    virtual unsigned int NumElementsRequired() = 0;
    virtual ModelParams Fit(std::vector<Element> elements) = 0;
    virtual bool IsInlier(const Element& element, float& score) = 0;
};


// Solver
template <class Element, class ModelParams>
class Solver
{
public:
    Solver(Model<Element, ModelParams>* model_, int max_iteration_)
        : m_model(model_), m_max_iteration(max_iteration_) {};

    ModelParams Solve(std::vector<Element> elements)
    {
        int size = elements.size();
        int num  = m_model->NumElementsRequired();
        std::vector<Element> selected;
        std::vector<bool> temp_is_inlier;
        temp_is_inlier.resize(size);

        int best_count = -1;
        float best_score = std::numeric_limits<float>::lowest();
        ModelParams best_params;
        std::vector<bool> final_is_inlier;
        final_is_inlier.resize(size);

        for (int i  = 0; i < m_max_iteration; i++)
        {
            selected.clear();

            // random sampling
            for (int n = 0; n < num; n++)
            {
                int idx = rand() % size;
                selected.emplace_back(elements[idx]);
            }

            // fit model params
            ModelParams params = m_model->Fit(selected);
            
            // count inliers (vs. outliers)
            int count = 0;
            float total_score = 0;
            for (int i = 0; i < size; i++)
            {
                float score = 0;
                if (m_model->IsInlier(elements[i], score))
                {
                    temp_is_inlier[i] = true;
                    total_score += score;
                    count++;
                }
                else
                {
                    temp_is_inlier[i] = false;
                }
            }

            // update best belief
            if (count > best_count)
            {
                best_params = params;
                best_count = count;
                for (int i = 0; i < size; i++)
                {
                    final_is_inlier[i] = temp_is_inlier[i];
                }
            }
        }

        // compute final params with inliers
        std::vector<Element> inliers;
        for (int i = 0; i < size; i++)
        {
            if (final_is_inlier[i] == true)
            {
                inliers.emplace_back(elements[i]);
            }
        }

        return m_model->Fit(inliers);
    };

private:
    int                             m_max_iteration;
    Model<Element, ModelParams>*    m_model;
};

} // RANSAC
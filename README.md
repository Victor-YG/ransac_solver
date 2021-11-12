# ransac_solver

Header-only ransac solver.

RANSAC is a very commonly used outlier rejection scheme in VO and SLAM. However simple, it is repetitive to write the routine for each new task. Instead, it can be generalized into having the solver class to execute the routine with a model class to provide necessary customization to each problem.

To use it, include the ransac_solver.h in your project.

To define a new model class, you need to:
1. Specify what is the individual element (Element). e.g., pixel pair, point pair, etc. It can be a std::pair of two items or an object containing two set of properties.
2. Specify how to describe the fitted model (ModelParams)
```
class YourModel : public RANSAC::Model<YourElementType, YourParamsType> {}
```

3. Implement the following functions:
```
// let solver know how many elements to sample to fit a model
unsigned int NumElementsRequired() {}
```

```
// how to fit the ModelParams given the list of selected elements
ModelParams Fit(std::vector<Element> elements) {}
```

```
// judge whether the input element is inlier or outlier.
bool IsInlier(const Element& element, float& score) {}
// you can also return a score used in the solver for judging how good the parameter fits the list of elements. If judging by counting inlier, simply set 1 for inlier and 0 for outlier.
// only score of inlier will be added to total score.
// if using loss metric, set score to be the negative of the loss.
```

4. In your main routine, 
  - create an instance of YourModel.
  - Create an instance of RANSAC::Solver.
  - Solve for the parameters.
```
YourModel model;
RANSAC::Solver<YourElementType, YourParamsType> solver(&model, max_iteration);
YourParamsType params = solver.Solve(vector_of_your_elements);
```

For a complete example, check the ./examples.

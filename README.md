# Predicting Remission Following CBT for Childhood Anxiety Disorders: A Machine Learning Approach

[![DOI](https://img.shields.io/badge/DOI-10.1017%2FS0033291724002654-blue)](https://doi.org/10.1017/S0033291724002654)

## About

This repository contains the code and analysis pipeline for our study on predicting anxiety disorder remission in youth following Cognitive Behavioral Therapy (CBT) using machine learning approaches.

## Publication

This code supports the following publication:

**Predicting remission following CBT for childhood anxiety disorders: a machine learning approach**  
*Psychological Medicine, Volume 54, Issue 16, December 2024, pp. 4612-4622*  
DOI: [10.1017/S0033291724002654](https://doi.org/10.1017/S0033291724002654)

## Abstract

**Background**  
The identification of predictors of treatment response is crucial for improving treatment outcome for children with anxiety disorders. Machine learning methods provide opportunities to identify combinations of factors that contribute to risk prediction models.

**Methods**  
A machine learning approach was applied to predict anxiety disorder remission in a large sample of 2114 anxious youth (5–18 years). Potential predictors included demographic, clinical, parental, and treatment variables with data obtained pre-treatment, post-treatment, and at least one follow-up.

**Results**  
All machine learning models performed similarly for remission outcomes, with AUC between 0.67 and 0.69. There was significant alignment between the factors that contributed to the models predicting two target outcomes: remission of all anxiety disorders and the primary anxiety disorder. Children who were older, had multiple anxiety disorders, comorbid depression, comorbid externalising disorders, received group treatment and therapy delivered by a more experienced therapist, and who had a parent with higher anxiety and depression symptoms, were more likely than other children to still meet criteria for anxiety disorders at the completion of therapy. In both models, the absence of a social anxiety disorder and being treated by a therapist with less experience contributed to the model predicting a higher likelihood of remission.

**Conclusions**  
These findings underscore the utility of prediction models that may indicate which children are more likely to remit or are more at risk of non-remission following CBT for childhood anxiety.

## Key Features

- Implementation of multiple machine learning models for predicting CBT remission
- Comprehensive feature importance analysis
- Cross-validation framework for model evaluation
- Visualization tools for interpreting model results

## Data

The analysis is based on a dataset of 2114 anxious youth (5-18 years) who received CBT treatment. Due to privacy concerns, the raw patient data is not included in this repository. Researchers interested in accessing the data should contact the corresponding author of the paper.

## Citation

If you use this code in your research, please cite our paper:

```
@article{cbt_anxiety_ml_2024,
  title={Predicting remission following CBT for childhood anxiety disorders: a machine learning approach},
  author={[Authors]},
  journal={Psychological Medicine},
  volume={54},
  number={16},
  pages={4612--4622},
  year={2024},
  publisher={Cambridge University Press},
  doi={10.1017/S0033291724002654}
}
```

## License

[MIT License](LICENSE)

## Contact

For questions or more information, please [open an issue](https://github.com/username/cbt-anxiety-ml-prediction/issues) or contact the corresponding author.

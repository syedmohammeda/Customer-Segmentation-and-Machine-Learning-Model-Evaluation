# Data

The clustering workflow downloads the publicly available Mall Customers dataset
on its first run and stores it as `Mall_Customers.csv` in this folder.

- Dataset: [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- Public CSV mirror used by the project: [Mall_Customers.csv](https://raw.githubusercontent.com/devotuoma/Mall-Customers-Segmentation/main/Mall_Customers.csv)
- Source fields: customer ID, gender, age, annual income (thousands), and spending score.
- The supervised workflow uses the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn; it does not download a second file.

If the CSV cannot be downloaded, the project creates a deterministic, clearly
identified synthetic customer sample so the complete pipeline can still run
offline. Replace that fallback with the original CSV to analyze the public data.
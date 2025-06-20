AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

| Class                                      | Precision | Recall | F1-Score | Support |
|--------------------------------------------|-----------|--------|----------|---------|
| **Apple**                                  |           |        |          |         |
| - Apple scab                               | 0.95      | 0.99   | 0.97     | 504     |
| - Black rot                                | 1.00      | 0.98   | 0.99     | 498     |
| - Cedar apple rust                         | 0.99      | 0.97   | 0.98     | 440     |
| - Healthy                                  | 0.96      | 0.94   | 0.95     | 502     |
| **Blueberry**                              |           |        |          |         |
| - Healthy                                  | 0.96      | 1.00   | 0.98     | 454     |
| **Cherry**                                 |           |        |          |         |
| - Powdery mildew                           | 0.97      | 0.99   | 0.98     | 422     |
| - Healthy                                  | 0.99      | 0.97   | 0.98     | 457     |
| **Corn (maize)**                           |           |        |          |         |
| - Cercospora leaf spot Gray leaf spot      | 0.99      | 0.93   | 0.96     | 411     |
| - Common rust                              | 0.99      | 1.00   | 1.00     | 478     |
| - Northern Leaf Blight                     | 0.95      | 0.99   | 0.97     | 478     |
| - Healthy                                  | 0.95      | 1.00   | 0.97     | 466     |
| **Grape**                                  |           |        |          |         |
| - Black rot                                | 0.97      | 0.98   | 0.97     | 472     |
| - Esca (Black Measles)                     | 0.99      | 0.98   | 0.99     | 480     |
| - Leaf blight (Isariopsis Leaf Spot)       | 0.96      | 1.00   | 0.98     | 431     |
| - Healthy                                  | 1.00      | 1.00   | 1.00     | 424     |
| **Orange**                                 |           |        |          |         |
| - Haunglongbing (Citrus greening)          | 0.99      | 0.98   | 0.99     | 503     |
| **Peach**                                  |           |        |          |         |
| - Bacterial spot                           | 0.94      | 0.98   | 0.96     | 460     |
| - Healthy                                  | 0.92      | 1.00   | 0.96     | 432     |
| **Pepper, bell**                           |           |        |          |         |
| - Bacterial spot                           | 0.98      | 0.97   | 0.98     | 479     |
| - Healthy                                  | 0.91      | 0.98   | 0.94     | 498     |
| **Potato**                                 |           |        |          |         |
| - Early blight                             | 0.97      | 1.00   | 0.98     | 486     |
| - Late blight                              | 0.95      | 0.97   | 0.96     | 486     |
| - Healthy                                  | 0.98      | 0.95   | 0.96     | 456     |
| **Raspberry**                              |           |        |          |         |
| - Healthy                                  | 0.99      | 0.97   | 0.98     | 446     |
| **Soybean**                                |           |        |          |         |
| - Healthy                                  | 1.00      | 0.97   | 0.98     | 506     |
| **Squash**                                 |           |        |          |         |
| - Powdery mildew                           | 1.00      | 0.96   | 0.98     | 434     |
| **Strawberry**                             |           |        |          |         |
| - Leaf scorch                              | 0.99      | 0.99   | 0.99     | 444     |
| - Healthy                                  | 1.00      | 0.99   | 0.99     | 456     |
| **Tomato**                                 |           |        |          |         |
| - Bacterial spot                           | 0.97      | 0.98   | 0.98     | 426     |
| - Early blight                             | 0.94      | 0.88   | 0.91     | 480     |
| - Late blight                              | 0.97      | 0.89   | 0.92     | 464     |
| - Leaf Mold                                | 0.94      | 0.98   | 0.96     | 471     |
| - Septoria leaf spot                       | 0.96      | 0.91   | 0.93     | 437     |
| - Spider mites Two-spotted spider mite     | 0.97      | 0.96   | 0.96     | 436     |
| - Target Spot                              | 0.92      | 0.93   | 0.93     | 458     |
| - Tomato Yellow Leaf Curl Virus            | 0.99      | 0.97   | 0.98     | 491     |
| - Tomato mosaic virus                      | 0.98      | 0.99   | 0.98     | 448     |
| - Healthy                                  | 0.99      | 0.96   | 0.97     | 482     |

### Overall Metrics
| Metric        | Value |
|---------------|-------|
| Accuracy      | 0.97  |
| Macro Avg     | 0.97  |
| Weighted Avg  | 0.97  |
| Total Support | 17596 |

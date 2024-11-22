import matplotlib.pyplot as plt
import json


metrics = json.load(open("checkpoints/bert_metrics.json"))
plt.plot(metrics["train_acc"])
plt.plot(metrics["test_acc"])
plt.show()

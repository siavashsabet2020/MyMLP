import matplotlib.pyplot as plt
from MLPClassifier import List_MseTrain, List_AccTrain, List_AccValid, List_MseValid


plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
plt.title('MSE')
plt.plot(List_MseTrain, label='MSE-Train')
plt.plot(List_MseValid, label='MSE-val')
plt.legend()
plt.xlabel('epochs')
plt.xlabel('MSE')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(List_AccTrain, label='ACC-Train')
plt.plot(List_AccValid, label='ACC-val')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

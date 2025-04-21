import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('cnn-mse.csv')

plt.figure(figsize=(10, 6))

plt.plot(df['1dayMSE'], label='1 Image MSE', marker='o')
plt.plot(df['7dayMSE'], label='7 Image MSE', marker='s')
plt.plot(df['30dayMSE'], label='30 Image MSE', marker='^')

plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('MSE over Time for Different Input Windows')
plt.legend()

plt.grid(True)

plt.tight_layout()
plt.savefig('cnn-mse.png')

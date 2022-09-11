import pandas as pd

df = pd.read_csv('HASC_DataPrj20110313/HascToolDataPrj/SampleData/1_stay/person101/HASC1161.csv')
df_all = pd.read_csv('all.csv')
span = 32
df = df.drop("19073.622081", axis=1)
for i in range((len(df)+span-1)//span):
    df[span*i:span*(i+1)].to_csv(f'dataset/test{i}.csv', index=False, header = False)
    df_all.loc[i] = [f'C:/Users/0414i/Documents/watanabeLab/syukatsu/sony/dataset/dataset/test{i}.csv', f'C:/Users/0414i/Documents/watanabeLab/syukatsu/sony/dataset/dataset/test{i}_.csv']
# df_all.to_csv('all_.csv', index=False)




import numpy as np

def extract_gravity():

    # // ローパスフィルターの係数(これは環境によって要調整。1に近づけるほど平滑化の度合いが大きくなる)
    filterCoefficient = 0.9
    lowpassValue = 0
    highpassValue = 0

    span = 32

    for j in range((len(df)+span-1)//span):
      newdf = df[span*j:span*(j+1)]
      highpassValue =np.empty_like(np.array(newdf.values))

      for i in range(span):

          # // ローパスフィルター(現在の値 = 係数 * ひとつ前の値 ＋ (1 - 係数) * センサの値)
          lowpassValue = lowpassValue * filterCoefficient + newdf.values[i] * (1 - filterCoefficient)
          # // ハイパスフィルター(センサの値 - ローパスフィルターの値)
          highpassValue[i] = newdf.values[i] - lowpassValue

      pd.DataFrame(highpassValue).to_csv(f'dataset/test{j}_.csv', index=False, header=False)


extract_gravity()
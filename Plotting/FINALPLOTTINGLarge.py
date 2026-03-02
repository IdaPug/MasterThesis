import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('DINOUNETFINAL_Large.txt')
df_Unet = pd.read_csv('DINOUNETFINAL.txt')

# print model in df_Unet
print("Models found in df_Unet:")
print(df_Unet['model'].unique())

# get the Vanilla UNet data
df_Vanilla = df_Unet[df_Unet['model'] == 'VanillaUNet_model']

# print 
print(df_Vanilla)

# merge with the original data to get the Vanilla UNet data in the same dataframe
df = pd.concat([df, df_Vanilla], ignore_index=True)

# print df
print("Combined DataFrame:")
print(df)

# print the models found
print("Models found in data:")
print(df['model'].unique())


# asign color and marker for each model
colors = {'AG_fullV2_constant': 'magenta', 'v1_constant': 'cyan', 'VanillaUNet_model': 'red','DinoEnvALT_constant': 'blue'}
markers = {'AG_fullV2_constant': 'o', 'v1_constant': 's', 'VanillaUNet_model': 'D','DinoEnvALT_constant': 'X'}
labels = {'AG_fullV2_constant': 'UNetDinoAttGate', 'v1_constant': 'UNetDino', 'VanillaUNet_model': 'Vanilla UNet','DinoEnvALT_constant': 'DinoEnc (vitb16)'}




#cross_data = df_alt2[df_alt2['model'] == 7]
#df = pd.concat([df, cross_data], ignore_index=True)

xticks = sorted(df['TrainSize'].unique())


plt.figure(figsize=(8,6))
for model in df['model'].unique():

    # skip if model is not in colors or markers
    if model not in colors or model not in markers:
        print(f"Model {model} not found in colors or markers, skipping.")
        continue

    # # also skip DinoEnv_constant
    # if model == 'DinoEnv_constant':
    #     print(f"Model {model} is DinoEnv_constant, skipping.")
    #     continue

  

    model_data = df[df['model'] == model]

    # sort model data by percent
    model_data = model_data.sort_values(by='TrainSize')
    print(f"Dice scores for model {model}:")
    print(model_data[['TrainSize', 'dice']])
    # plot with log scale on x axis
    plt.xscale('log')
    plt.plot(model_data['TrainSize'], model_data['dice'], marker=markers[model], color=colors[model], label=labels[model])

# set xtickslabels
plt.xticks(xticks, labels=[str(x) for x in xticks])

plt.xlabel('Training Set Size (%)')
plt.ylabel('Dice Score')
plt.title('DINO Model Performance vs Training Set Size')
plt.legend()
plt.grid(True)
# save with minimal whitespace around the plot
plt.savefig('LargeTrainingDataFinalPerformanceVIT_B.png', bbox_inches='tight')
plt.savefig('LargeTrainingDataFinalPerformanceVIT_B.pdf', bbox_inches='tight')




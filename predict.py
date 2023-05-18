import dataloaders.datasetnormal
import torch
#
#

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device=torch.device('cpu')
val_data_dir = "dataset/store/validation128mel.pkl"
test_loader=dataloaders.datasetnormal.fetch_dataloader(val_data_dir,32,1)
for batch_idx,data in enumerate(test_loader):
    predict=data[0].to(device)
    target=data[1].squeeze(1).to(device)

# pred_x=[]
# pred_y=[]
#
# for batch_idx,data in enumerate(val_loader):
#         input=data[0]
#         target=data[1].squeeze(1)
#         pred_x.append(input)
#         pred_y.append(target)
# print(pred_x)
# print(pred_y)
# predict_data_path = os.path.join("dataset/meta/CSC4_val.csv")
# predict_audio_dir = os.path.join("dataset/audio_val")
# predict_meta_date = pd.read_csv(predict_data_path)
#
# pred_x = []
# preddata = ef.extract_features(predict_meta_date, predict_audio_dir)
# # print(preddata)
# for data in preddata:
#     # print(data['target'])
#     pred_x.append(data['audio'])
#     print(np.array(pred_x).shape)
# # print(len(pred_x))
# pred = None
# path = os.path.join('best_cough.pkl')
# # print(path)
#
# # for path in ["checkpoint_dir_dense(P)/model_best_1_v2.pth.tar"]:
# model = torch.load(path, map_location='cpu')
#
# if pred is None:
#         pred = model.predict(pred_x)
# else:
#         pred += model.predict(pred_x)

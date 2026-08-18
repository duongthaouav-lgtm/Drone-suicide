from ultralytics import YOLO

model = YOLO("./weights/best_ncnn_model/model.param", task="detect")

print(type(model.model))
# In toàn bộ info
# print(model.model)
# print(model.names)




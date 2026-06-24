import torch
import torch.nn as nn
import timm 

NUM_CLASSES = {'race': 7, 'gender': 2, 'age': 9}

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # 4 khối Convolution-BatchNorm-ReLU-MaxPooling 
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = 256

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1) # Chuyển tensor thành vector đặc trưng
        return x


# 2. MÔ HÌNH MULTI-TASK LEARNING
# Kết hợp Backbone với 3 nhánh phân loại độc lập:
# - Race Classification
# - Gender Classification
# - Age Classification
class FairFaceModel(nn.Module):
    def __init__(self, num_classes: dict, timm_id: str, dropout_rate: float, name: str = '', img_size: int = 224):
        super().__init__()
        self.model_name = name or timm_id
        self.is_custom = (timm_id == 'custom_cnn')

        # Khởi tạo Backbone và lấy feat_dim
        if self.is_custom:
            self.backbone = CustomCNN()
            feat_dim = self.backbone.num_features
        else:
            # Sử dụng backbone pretrained từ thư viện timm
            self.backbone = timm.create_model(timm_id, pretrained=True, num_classes=0, global_pool='avg')
            dummy_input = torch.randn(1, 3, img_size, img_size)
            with torch.no_grad():
                feat = self.backbone(dummy_input)
            feat_dim = feat.shape[1]

        # Khởi tạo 3 Heads
        self.race_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes['race'])
        )

        self.gender_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes['gender'])
        )

        self.age_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes['age'])
        )

        # Khởi tạo trọng số cho các lớp Fully Connected
        self._init_weights()

    # Khởi tạo trọng số cho các Classification Heads
    def _init_weights(self):
        for head in [self.race_head, self.gender_head, self.age_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Trích xuất đặc trưng (Feature extraction)
        feat = self.backbone(x)

        # Dự đoán đồng thời 3 tác vụ
        race_out = self.race_head(feat)
        gender_out = self.gender_head(feat)
        age_out = self.age_head(feat)

        return race_out, gender_out, age_out

    # HỖ TRỢ TRANSFER LEARNING
    # Freeze/Unfreeze backbone trong quá trình huấn luyện
    def freeze_backbone(self):
        if not self.is_custom:
            for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        if not self.is_custom:
            for p in self.backbone.parameters(): p.requires_grad = True

    # Thống kê số lượng tham số của mô hình
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import json

# Tạo dữ liệu mẫu cho training với công thức chi tiết
data = {
    'dish_name_vn': [
        'Thịt kho trứng', 'Rau xào tỏi', 'Trứng chiên', 'Canh rau củ',
        'Thịt bò xào rau', 'Cá chiên', 'Súp gà', 'Mì xào hải sản',
        'Cơm rang', 'Gà nướng', 'Bún chả', 'Phở bò', 'Bánh xèo',
        'Chả giò', 'Bò lúc lắc', 'Tôm sốt chua ngọt', 'Cơm gà',
        'Bò kho', 'Canh chua cá', 'Rau muống xào tỏi', 'Đậu sốt cà chua',
        'Thịt heo luộc', 'Cá kho tộ', 'Gà xào sả ớt', 'Bò xào cần tỏi',
        'Tôm rim', 'Trứng vịt lộn', 'Lẩu thái', 'Bún bò huế', 'Hủ tiếu'
    ],
    'ingredients': [
        ['pork', 'egg', 'soy_sauce', 'sugar', 'pepper'],
        ['vegetable', 'garlic', 'oil', 'salt'],
        ['egg', 'oil', 'salt'],
        ['carrot', 'potato', 'tomato', 'onion'],
        ['beef', 'vegetable', 'garlic', 'soy_sauce'],
        ['fish', 'oil', 'salt', 'pepper'],
        ['chicken', 'carrot', 'potato', 'onion'],
        ['noodle', 'shrimp', 'squid', 'vegetable', 'soy_sauce'],
        ['rice', 'egg', 'vegetable', 'soy_sauce'],
        ['chicken', 'honey', 'soy_sauce', 'garlic'],
        ['pork', 'noodle', 'vegetable', 'fish_sauce'],
        ['beef', 'rice_noodle', 'onion', 'spices'],
        ['flour', 'shrimp', 'pork', 'bean_sprout'],
        ['pork', 'shrimp', 'vegetable', 'rice_paper'],
        ['beef', 'pepper', 'garlic', 'soy_sauce'],
        ['shrimp', 'tomato', 'pineapple', 'sugar'],
        ['chicken', 'rice', 'ginger', 'fish_sauce'],
        ['beef', 'carrot', 'potato', 'spices'],
        ['fish', 'tomato', 'pineapple', 'tamarind'],
        ['water_spinach', 'garlic', 'oil', 'fish_sauce'],
        ['tofu', 'tomato', 'onion', 'soy_sauce'],
        ['pork', 'salt', 'ginger'],
        ['fish', 'sugar', 'fish_sauce', 'pepper'],
        ['chicken', 'lemongrass', 'chili', 'fish_sauce'],
        ['beef', 'celery', 'garlic', 'oyster_sauce'],
        ['shrimp', 'sugar', 'fish_sauce', 'pepper'],
        ['duck_egg', 'salt', 'pepper'],
        ['shrimp', 'mushroom', 'tomato', 'lemongrass'],
        ['beef', 'rice_noodle', 'lemongrass', 'chili'],
        ['pork', 'noodle', 'shrimp', 'onion']
    ],
    'cooking_steps': [
        [
            "Rửa sạch thịt heo, cắt miếng vừa ăn",
            "Ướp thịt với nước mắm, đường, tiêu trong 15 phút",
            "Luộc trứng chín, bóc vỏ",
            "Kho thịt với nước dừa trong 30 phút",
            "Thêm trứng vào kho thêm 10 phút",
            "Nêm nếm vừa ăn, thêm hành lá"
        ],
        [
            "Rửa sạch rau, để ráo",
            "Băm nhỏ tỏi",
            "Đun nóng dầu, phi tỏi thơm",
            "Xào rau với lửa lớn trong 3 phút",
            "Nêm muối vừa ăn",
            "Cho ra đĩa, ăn nóng"
        ],
        [
            "Đập trứng vào bát, đánh đều",
            "Thêm chút muối vào trứng",
            "Đun nóng dầu trong chảo",
            "Đổ trứng vào chảo, chiên vàng hai mặt",
            "Cho ra đĩa, ăn với cơm"
        ],
        [
            "Rửa sạch rau củ, cắt miếng vừa ăn",
            "Đun sôi 1.5 lít nước",
            "Cho rau củ cứng vào trước (cà rốt, khoai tây)",
            "Nấu 10 phút, sau đó cho cà chua, hành tây",
            "Nêm muối, hạt nêm vừa ăn",
            "Nấu thêm 5 phút, tắt bếp"
        ],
        [
            "Thái thịt bò mỏng, ướp với tỏi băm và nước tương",
            "Rửa sạch rau, cắt khúc",
            "Đun nóng dầu, xào thịt bò chín tái",
            "Cho rau vào xào chung",
            "Nêm nếm gia vị vừa ăn",
            "Xào thêm 2 phút, cho ra đĩa"
        ],
        [
            "Làm sạch cá, để ráo",
            "Ướp cá với muối, tiêu, chanh trong 10 phút",
            "Đun nóng dầu trong chảo",
            "Chiên cá vàng đều hai mặt",
            "Cho ra đĩa có lót giấy thấm dầu",
            "Ăn kèm nước mắm chua ngọt"
        ],
        [
            "Rửa sạch gà, chặt miếng vừa ăn",
            "Đun sôi nước, cho gà vào luộc 10 phút",
            "Vớt gà ra, giữ nước dùng",
            "Cho cà rốt, khoai tây, hành tây vào nấu",
            "Nấu 15 phút cho rau củ mềm",
            "Nêm gia vị, thêm gà vào đun 5 phút"
        ],
        [
            "Luộc mì chín, vớt ra để ráo",
            "Rửa sạch hải sản",
            "Đun nóng dầu, xào tỏi thơm",
            "Cho hải sản vào xào chín",
            "Thêm rau củ vào xào",
            "Cho mì vào, đảo đều với nước tương"
        ],
        [
            "Cho dầu vào chảo, đun nóng",
            "Cho trứng vào, đảo nhanh tay",
            "Thêm cơm nguội vào đảo đều",
            "Thêm rau củ tùy thích",
            "Nêm nước tương, muối vừa ăn",
            "Đảo đều 5-7 phút, cho ra đĩa"
        ],
        [
            "Rửa sạch gà, để ráo",
            "Ướp gà với mật ong, nước tương, tỏi 30 phút",
            "Làm nóng lò nướng 200°C",
            "Nướng gà 25-30 phút, lật mặt giữa chừng",
            "Kiểm tra gà chín bằng cách chọc đũa",
            "Lấy ra, cắt miếng, trang trí"
        ],
        [
            "Thịt heo xay ướp gia vị, nặn thành viên",
            "Nướng chả trên than hoa hoặc chảo",
            "Pha nước mắm chua ngọt",
            "Luộc bún, để ráo",
            "Rau sống rửa sạch",
            "Bày bún, chả, rau ra đĩa, chan nước mắm"
        ],
        [
            "Hầm xương bò lấy nước dùng",
            "Thái thịt bò mỏng",
            "Luộc bánh phở",
            "Cho bánh phở vào tô",
            "Xếp thịt bò lên trên",
            "Chan nước dùng nóng, thêm hành, rau thơm"
        ],
        [
            "Pha bột bánh xèo với nước cốt dừa",
            "Nhân bánh: tôm, thịt, giá đỗ",
            "Đổ bột vào chảo nóng, tráng mỏng",
            "Cho nhân vào giữa bánh",
            "Gập bánh lại khi chín vàng",
            "Ăn kèm rau sống và nước mắm"
        ],
        [
            "Trộn nhân: thịt heo xay, tôm, rau củ",
            "Ngâm bánh tráng cho mềm",
            "Cuộn chả giò chặt tay",
            "Chiên vàng đều trong dầu nóng",
            "Vớt ra để ráo dầu",
            "Ăn kèm rau sống và nước mắm chua ngọt"
        ],
        [
            "Thái thịt bò thành khối vuông",
            "Ướp thịt với tiêu, tỏi, nước tương",
            "Đun nóng dầu, xào thịt bò chín tái",
            "Thêm hành tây, ớt chuông",
            "Xào nhanh trên lửa lớn",
            "Cho ra đĩa, ăn nóng"
        ],
        [
            "Rửa sạch tôm, để ráo",
            "Cắt cà chua, dứa miếng vừa ăn",
            "Pha sốt chua ngọt: cà chua, dứa, đường, giấm",
            "Xào tôm chín hồng",
            "Cho sốt vào đun sôi",
            "Nêm nếm, thêm hành lá"
        ],
        [
            "Luộc gà với gừng",
            "Xé thịt gà thành miếng vừa ăn",
            "Nấu cơm với nước luộc gà",
            "Pha nước mắm gừng",
            "Bày cơm, gà ra đĩa",
            "Rắc hành phi, ăn kèm nước mắm"
        ],
        [
            "Thái thịt bò miếng vừa ăn",
            "Ướp thịt với gia vị kho",
            "Đun nóng dầu, xào thịt săn lại",
            "Thêm nước, cà rốt, khoai tây",
            "Kho nhỏ lửa 1.5-2 giờ",
            "Nêm nếm, thêm hành ngò"
        ],
        [
            "Làm sạch cá",
            "Nấu nước sôi, cho me vào khuấy tan",
            "Thả cá vào nồi",
            "Cho cà chua, dứa, đậu bắp",
            "Nêm mắm muối, đường, ớt",
            "Nấu 15 phút, thêm rau om, ngò gai"
        ],
        [
            "Rau muống rửa sạch, cắt khúc",
            "Băm tỏi nhỏ",
            "Đun nóng dầu, phi tỏi thơm",
            "Cho rau vào xào nhanh lửa lớn",
            "Nêm nước mắm, bột nêm",
            "Xào chín tái, cho ra đĩa"
        ],
        [
            "Cắt đậu phụ thành miếng vuông",
            "Chiên đậu vàng nhẹ",
            "Xào cà chua với hành tây",
            "Thêm nước, nước tương, đường",
            "Cho đậu vào sốt 10 phút",
            "Thêm hành lá, tắt bếp"
        ],
        [
            "Rửa sạch thịt heo",
            "Đun sôi nước với gừng",
            "Cho thịt vào luộc 20-25 phút",
            "Vớt thịt ra, thái lát",
            "Pha nước mắm gừng",
            "Ăn kèm rau sống và bún"
        ],
        [
            "Làm sạch cá",
            "Ướp cá với nước mắm, đường, tiêu",
            "Xếp cá vào nồi đất",
            "Thêm nước dừa, nước mắm",
            "Kho nhỏ lửa 30-40 phút",
            "Thêm ớt, hành lá"
        ],
        [
            "Băm nhỏ sả, ớt",
            "Ướp gà với sả, ớt, nước mắm",
            "Đun nóng dầu, xào gà chín vàng",
            "Thêm chút nước, đậy vung",
            "Nấu thêm 10 phút",
            "Nêm nếm, thêm hành lá"
        ],
        [
            "Thái thịt bò mỏng",
            "Cần tây cắt khúc",
            "Ướp thịt với tỏi, dầu hào",
            "Xào thịt bò chín tái, vớt ra",
            "Xào cần tây với tỏi",
            "Cho thịt bò vào đảo đều, tắt bếp"
        ],
        [
            "Rửa sạch tôm",
            "Ướp tôm với đường, nước mắm, tiêu",
            "Đun nhỏ lửa với ít nước",
            "Rim đến khi nước cạn, tôm săn lại",
            "Thêm tỏi phi nếu thích",
            "Để nguội, ăn với cơm"
        ],
        [
            "Rửa sạch trứng vịt lộn",
            "Luộc trứng 20-25 phút",
            "Bóc vỏ, để nguyên quả",
            "Ăn kèm rau răm, gừng thái sợi",
            "Chấm muối tiêu chanh",
            "Uống kèm trà nóng"
        ],
        [
            "Chuẩn bị nước lẩu: nấu xương, thêm sả, me",
            "Thái các loại rau củ, nấm",
            "Sơ chế hải sản",
            "Đun sôi nước lẩu",
            "Nhúng nguyên liệu ăn kèm",
            "Chấm nước mắm lẩu thái"
        ],
        [
            "Hầm xương bò lấy nước dùng",
            "Thêm sả, ớt, mắm ruốc",
            "Luộc bún bò",
            "Thái thịt bò chín và tái",
            "Bày bún vào tô, xếp thịt",
            "Chan nước dùng nóng, thêm hành, rau"
        ],
        [
            "Nấu nước dùng từ xương heo",
            "Luộc hủ tiếu, để ráo",
            "Xào thịt heo, tôm với hành tây",
            "Cho hủ tiếu vào xào",
            "Nêm nước tương, gia vị",
            "Cho ra đĩa, rắc hành lá"
        ]
    ],
    'cooking_time': [
        "45 phút", "10 phút", "5 phút", "25 phút",
        "20 phút", "15 phút", "30 phút", "25 phút",
        "15 phút", "50 phút", "40 phút", "3 giờ",
        "30 phút", "40 phút", "20 phút", "25 phút",
        "1 giờ", "2 giờ", "30 phút", "10 phút",
        "20 phút", "30 phút", "1 giờ", "25 phút",
        "20 phút", "30 phút", "30 phút", "45 phút",
        "4 giờ", "40 phút"
    ],
    'difficulty': [
        "Trung bình", "Dễ", "Rất dễ", "Dễ",
        "Trung bình", "Dễ", "Trung bình", "Trung bình",
        "Dễ", "Trung bình", "Khó", "Khó",
        "Trung bình", "Trung bình", "Dễ", "Trung bình",
        "Dễ", "Trung bình", "Trung bình", "Dễ",
        "Dễ", "Dễ", "Trung bình", "Trung bình",
        "Trung bình", "Dễ", "Dễ", "Trung bình",
        "Khó", "Trung bình"
    ],
    'category': [
        "Món mặn", "Món rau", "Món trứng", "Món canh",
        "Món mặn", "Món cá", "Món súp", "Món mì",
        "Món cơm", "Món gà", "Món bún", "Món phở",
        "Món bánh", "Món chiên", "Món bò", "Món tôm",
        "Món cơm", "Món kho", "Món canh", "Món rau",
        "Món chay", "Món luộc", "Món kho", "Món gà",
        "Món bò", "Món tôm", "Món trứng", "Món lẩu",
        "Món bún", "Món mì"
    ]
}

# Tạo DataFrame
df = pd.DataFrame(data)

# Tạo bộ từ vựng nguyên liệu
all_ingredients = set()
for ingredients in df['ingredients']:
    all_ingredients.update(ingredients)

# Mã hóa nguyên liệu thành vector nhị phân
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['ingredients'])
y = np.arange(len(df))  # Mỗi món ăn là một lớp

# Lưu bộ mã hóa
joblib.dump(mlb, 'ingredients_encoder.pkl')

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Đánh giá mô hình
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)

print(f"Training Accuracy: {train_score:.2%}")
print(f"Testing Accuracy: {test_score:.2%}")

# Lưu mô hình
joblib.dump(rf_model, 'random_forest_model.pkl')

# Lưu thông tin món ăn đầy đủ
recipes_info = df.to_dict('records')
with open('recipes_info.json', 'w', encoding='utf-8') as f:
    json.dump(recipes_info, f, ensure_ascii=False, indent=2)

print("Đã huấn luyện và lưu mô hình Random Forest thành công!")
print(f"Số lượng nguyên liệu: {len(all_ingredients)}")
print(f"Số lượng món ăn: {len(df)}")
print(f"Đã thêm công thức nấu ăn chi tiết cho {len(df)} món ăn!")
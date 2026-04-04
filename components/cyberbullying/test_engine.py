from core.inference import predict_meme

# change this to a REAL image path from your dataset
TEST_IMAGE = r"D:\Research\project\test_samples\non_bullying\03798.png"

# optional caption
TEST_CAPTION = ""

result = predict_meme(TEST_IMAGE, TEST_CAPTION)

print("\n===== PREDICTION RESULT =====")
for key, value in result.items():
    print(f"{key}: {value}")


   
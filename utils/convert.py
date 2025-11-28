import tensorflow as tf

# Load your keras model (with compile=False to avoid optimizer issues)
model = tf.keras.models.load_model('Models/best_model.keras', compile=False)

# Save as H5
model.save('Models/best_model.h5')

print(f"Converted! File size: {os.path.getsize('Models/best_model.h5') / (1024*1024):.2f} MB")
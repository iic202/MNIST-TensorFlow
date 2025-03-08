import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_data():
    """Load and preprocess the MNIST dataset."""
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Normalize the images to [0, 1] range
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # Reshape the images to flatten them (for sequential NN, not CNN)
    train_images = train_images.reshape(-1, 28*28)
    test_images = test_images.reshape(-1, 28*28)
    
    # Print dataset info
    print(f"Training data shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    return (train_images, train_labels), (test_images, test_labels)

def build_model():
    """Build a sequential neural network with 1 hidden layer and 1 output layer."""
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(28*28,)),  # Hidden layer with 128 nodes
        layers.Dense(10, activation='softmax')                       # Output layer with 10 nodes
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

def train_model(model, train_images, train_labels, test_images, test_labels):
    """Train the model and return training history."""
    history = model.fit(
        train_images, train_labels, 
        epochs=10, 
        validation_data=(test_images, test_labels)
    )
    return history

def evaluate_model(model, test_images, test_labels):
    """Evaluate the model on test data."""
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    return test_loss, test_acc

def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

    plt.savefig("figures/training_history.png")

def plot_conffusion_matrix(model, test_images, test_labels):
    """Plot the confusion matrix for the model."""
    # Make predictions
    predictions = model.predict(test_images)
    
    # Plot the confusion matrix
    confusion_matrix = tf.math.confusion_matrix(test_labels, np.argmax(predictions, axis=1))
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig("figures/confusion_matrix.png")

def plot_image(i, predictions_array, true_label, img):
    """Plot a test image along with its prediction."""
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    # Reshape back to 28x28 for visualization
    plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
        
    plt.xlabel(f"Pred: {predicted_label} ({100*np.max(predictions_array):0.1f}%)\nTrue: {true_label}", 
               color=color)

def visualize_predictions(model, test_images, test_labels):
    """Visualize predictions on test examples."""
    # Make predictions
    predictions = model.predict(test_images[:10])
    
    # Plot the first 5 test images with their predictions
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plot_image(i, predictions[i], test_labels, test_images)
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess data
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    
    # Build model
    model = build_model()
    
    # Train model
    history = train_model(model, train_images, train_labels, test_images, test_labels)
    
    # Evaluate model
    evaluate_model(model, test_images, test_labels)
    
    # Plot training history
    plot_training_history(history)

    # Plot confusion matrix
    plot_conffusion_matrix(model, test_images, test_labels)
    
    # Visualize predictions
    visualize_predictions(model, test_images, test_labels)

if __name__ == "__main__":
    main()
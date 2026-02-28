import numpy as np
import cupy as cp
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import json

IMAGE_SIZE = 96

def get_gpu_name():
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        return props['name'].decode('utf-8')
    except:
        return "Unknown GPU"

# loads images from filepath and returns as array
def load_image(filepath):
    # open with pillow
    img = Image.open(filepath)

    # force RGB
    img = img.convert('RGB')

    # pad to square
    width, height = img.size
    new_size = max(width, height)

    # create black background
    new_img =   Image.new("RGB", (new_size, new_size), (0, 0, 0))

    # paste original image in center
    paste_x = (new_size - width) // 2
    paste_y = (new_size - height) // 2
    new_img.paste(img, (paste_x, paste_y))

    # resize to target resolution
    new_img = new_img.resize((IMAGE_SIZE, IMAGE_SIZE))

    # return as array (image size, image size, 3)
    return np.array(new_img)


def load_data(data_dir):
    print("Loading data...")
    images = []
    labels = []

    # define class mapping
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_names.sort()

    print(f"Found {len(class_names)} species: {class_names}")

    for class_index, class_name in enumerate(class_names):
        folder_path = os.path.join(data_dir, class_name)

        # check if folder exists
        if not os.path.isdir(folder_path):
            print(f"WARN: Folder {folder_path} not found.")
            continue
        
        # iterate over every file in folder
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(folder_path, filename)

                img = load_image(filepath)

                # more channels
                img = img.transpose(2, 0, 1)
                
                images.append(img)
                labels.append(class_index)
    
    return np.array(images), np.array(labels), class_names


class ConvolutionLayer:
    def __init__(self, num_filters, filter_size, input_channels=1, padding=1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.padding = padding

        # initialize filters (num_filters, channels, H, W)
        scale = cp.sqrt(2.0 / (input_channels * filter_size * filter_size))
        self.filters = cp.random.randn(num_filters, input_channels, filter_size, filter_size).astype(cp.float32) * scale
        self.biases = cp.zeros(num_filters, dtype=cp.float32)
        
    # transforms image into group of filtered images
    def forward(self, input):
        # input shaped expected (batch, channels, height, width)
        self.last_input = input
        n_filters, d_filter, h_filter, w_filter = self.filters.shape
        n_x, d_x, h_x, w_x = input.shape

        # vectorize input (image to column)
        self.x_cols = im2col_indices(input, h_filter, w_filter, padding=self.padding, stride=1)

        # flatten filters
        # (num_filters, chanels * h * w)
        self.w_row = self.filters.reshape(self.num_filters, -1)

        # matrix multiplication (uses GPU)
        # result (num_filters, number_of_patches)
        out = cp.dot(self.w_row, self.x_cols) + self.biases.reshape(-1, 1)

        # reshape back to image (batch, num_filters, h_out, w_out)
        h_out = (h_x + 2 * self.padding - h_filter) // 1 + 1
        w_out = (w_x + 2 * self.padding - w_filter) // 1 + 1

        out = out.reshape(self.num_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)

        return out

    # backprop
    def backward(self, d_L_d_out, learning_rate):
        # when i coded this, only god and i knew what it was did exactly
        # now only god knows

        # d_L_d_out shape (batch, num_filters, h_out, w_out)
        n_filter, d_filter, h_filter, w_filter = self.filters.shape

        # reshape gradient for matrix multiply
        # (num_filters, batch * h_out, w_out)
        dout_reshaped = d_L_d_out.transpose(1, 2, 3, 0).reshape(n_filter, -1)

        # calculate filter gradients
        # (num_filters, inputs) dot (inputs, patches)^T
        d_L_d_filters = cp.dot(dout_reshaped, self.x_cols.T)
        d_L_d_filters = d_L_d_filters.reshape(self.filters.shape)

        # calculate bias gradients
        d_L_d_biases = cp.sum(dout_reshaped, axis=1)

        # calculate input gradients
        d_cols = cp.dot(self.w_row.T, dout_reshaped)

        # un-vectorize (turn cols back to image by col2im)
        d_L_d_input = col2im_indices(d_cols, self.last_input.shape, h_filter, w_filter, padding=self.padding)

        # update weights
        self.filters -= learning_rate * d_L_d_filters
        self.biases -= learning_rate * d_L_d_biases

        return d_L_d_input


# applied directly after the convolution
class ReLULayer:

    # simple filter, if less than 0, make zero, else ignore
    def forward(self, input):
        # saves inputs for backpass later
        self.last_input = input

        # compares elements and returns max of (0, input)
        return cp.maximum(0, input)
    
    # backprop
    def backward(self, d_L_d_out):
        # create copy so incoming gradient is not modified
        d_L_d_input = d_L_d_out.copy()

        # where input is <= 0, gradient is 0
        d_L_d_input[self.last_input <= 0] = 0

        return d_L_d_input


class MaxPoolingLayer:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
    
    # performs forward pass of max pooling layer using given input
    def forward(self, input):
        # input (batch, channels, height, width)
        self.last_input = input
        n, c, h, w = input.shape

        # reshape to (batch, channels, new_h, pool_size, new_w, pool_size)
        # groups 2x2 blocks together
        reshaped = input.reshape(n, c, h // self.pool_size, self.pool_size, w // self.pool_size, self.pool_size)

        # take max along pool dimensions
        out = reshaped.max(axis=(3, 5))
        return out

    # backprop
    def backward(self, d_L_d_out):
        # d_L_d_out is gradient coming from softmax layer
        
        # repeat gradient to match 2x2 blocks
        # (batch, c, h_out, w_out) -> (batch, C, h_out, 2, w_out, 2)
        d_L_d_out_expanded = d_L_d_out[:, :, :, cp.newaxis, :, cp.newaxis]
        d_L_d_out_expanded = cp.repeat(d_L_d_out_expanded, self.pool_size, axis=3)
        d_L_d_out_expanded = cp.repeat(d_L_d_out_expanded, self.pool_size, axis=5)

        # reshape to match original input structure
        d_L_d_out_expanded = d_L_d_out_expanded.reshape(self.last_input.shape)

        # create mask of which pixels were max
        # recalculate max broadcasted to original shape
        n, c, h, w = self.last_input.shape
        reshaped = self.last_input.reshape(n, c, h // self.pool_size, self.pool_size, w // self.pool_size, self.pool_size)
        max_vals = reshaped.max(axis=(3, 5), keepdims=True)
        # broadcast max values back to 2x2 blocks
        max_vals = cp.repeat(cp.repeat(max_vals, self.pool_size, axis=3), self.pool_size, axis=5)
        max_vals = max_vals.reshape(self.last_input.shape)

        mask = (self.last_input == max_vals)

        # multiply gradient by sum
        return d_L_d_out_expanded * mask



class SoftmaxLayer:
    def __init__(self, input_len, nodes):
        # weights (features, classes)
        self.weights = cp.random.randn(input_len, nodes).astype(cp.float32) / cp.sqrt(input_len)
        self.biases = cp.zeros(nodes, dtype=cp.float32)
    
    # flattens the input and calculates final probabilities
    def forward(self, input):
        self.last_input_shape = input.shape

        # flatten (batch, channels, h, w) -> (batch, features)
        input_flattened = input.reshape(input.shape[0], -1)
        self.last_input = input_flattened

        # matrix multiplication (batch * features) dot (features * classes)
        totals = cp.dot(input_flattened, self.weights) + self.biases

        # softmax, subtract max to keep stable
        totals_max = cp.max(totals, axis=1, keepdims=True)
        exp = cp.exp(totals - totals_max)
        return exp / cp.sum(exp, axis=1, keepdims=True)
    
    # backprop pass
    def backward(self, d_L_d_out, learning_rate):
        # d_L_d_out is (batch, classes)

        # gradient of weights
        d_L_d_w = cp.dot(self.last_input.T, d_L_d_out)

        # gradient of biases
        d_L_d_b = cp.sum(d_L_d_out, axis=0)

        # gradient of input
        d_L_d_inputs = cp.dot(d_L_d_out, self.weights.T)

        # update weights
        self.weights -= learning_rate * d_L_d_w
        self.biases -= learning_rate * d_L_d_b

        return d_L_d_inputs.reshape(self.last_input_shape)


class DropoutLayer:
    def __init__(self, probability=0.25):
        self.probability = probability
        self.mask = None
    
    def forward(self, input, is_training=True):
        self.last_input = input
        if is_training:
            # create a mask of 1 and 0
            self.mask = (cp.random.rand(*input.shape) > self.probability) / (1.0 - self.probability)
            return input * self.mask
        else:
            return input
    
    def backward(self, d_L_d_out):
        # pass gradient where neurons were not dropped
        return d_L_d_out * self.mask



def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # N = batch size, C = channels, H = height, W = width
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_width) % stride == 0

    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    # create vectors for filter positions
    i0 = cp.repeat(cp.arange(field_height), field_width)
    i0 = cp.repeat(i0, C)
    i1 = stride * cp.repeat(cp.arange(out_height), out_width)
    j0 = cp.tile(cp.arange(field_width), field_height * C)
    j1 = stride * cp.tile(cp.arange(out_width), out_height)

    # create index matrices for broadcasting
    i = j0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = cp.repeat(cp.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    # pad input (batch, channel, height, width)
    x_padded = cp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    # indexing: select all pixels for all patches at once
    cols = x_padded[:, k, i, j]

    # reshape for matrix multiplication
    C = x.shape[1]
    # channels = filter_height * filter_width
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = cp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)

    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    # add gradients back to original pixel location
    cp.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def export_to_json(conv_layers, softmax, dir='docs/model/model_weights.json'):
    # extract weights
    data = {
        "conv": [
            {"weights": l.filters.tolist(), "biases": l.biases.tolist()} 
            for l in conv_layers
        ],
        "softmax": {
            "weights": softmax.weights.tolist(),
            "biases": softmax.biases.tolist()
        }
    }
    with open(dir, 'w') as f:
        json.dump(data, f)
    print(f"Model exported for JS: {dir}")


def train(images, labels, num_classes, num_epochs=10, lr=0.005, batch_size=64):
    # setup network
    
    # block 1
    conv1 = ConvolutionLayer(32, 3, input_channels=3, padding=1)
    relu1 = ReLULayer()
    pool1 = MaxPoolingLayer(2)

    # block 2
    conv2 = ConvolutionLayer(64, 3, input_channels=32, padding=1)
    relu2 = ReLULayer()
    pool2 = MaxPoolingLayer(2)

    # block 3
    conv3 = ConvolutionLayer(64, 3, input_channels=64, padding=1)
    relu3 = ReLULayer()
    pool3 = MaxPoolingLayer(2)

    # block 4
    conv4 = ConvolutionLayer(128, 3, input_channels=64, padding=1)
    relu4 = ReLULayer()
    pool4 = MaxPoolingLayer(2)

    # dropout before classification to prevent overfitting
    dropout = DropoutLayer(0.25)

    # softmax
    softmax = SoftmaxLayer(6 * 6 * 128, num_classes)

    print("Pushing dataset to GPU VRAM...")
    images_gpu = cp.asarray(images, dtype=cp.float32)
    labels_gpu = cp.asarray(labels, dtype=cp.int32)

    # training loop
    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch + 1} ---")

        # shuffle data each epoch
        permutation = np.random.permutation(len(images_gpu))
        images_gpu = images_gpu[permutation]
        labels_gpu = labels_gpu[permutation]

        loss = 0
        num_correct = 0

        for i in range(0, len(images), batch_size):
            
            # handle the final batch if it's smaller than batch_size
            x_batch = images_gpu[i : i + batch_size]
            y_batch = labels_gpu[i : i + batch_size]

            indices_to_flip = cp.random.rand(len(x_batch)) > 0.5

            x_batch_aug = x_batch.copy()
            x_batch_aug[indices_to_flip] = x_batch_aug[indices_to_flip, :, :, ::-1]

            # forward pass
            out = conv1.forward(x_batch_aug)
            out = relu1.forward(out)
            out = pool1.forward(out)
            
            out = conv2.forward(out)
            out = relu2.forward(out)
            out = pool2.forward(out)
            
            out = conv3.forward(out)
            out = relu3.forward(out)
            out = pool3.forward(out)
            
            out = conv4.forward(out)
            out = relu4.forward(out)
            out = pool4.forward(out)

            out = dropout.forward(out, is_training=True)
            
            probs = softmax.forward(out)

            # calculate loss and accuracy
            log_probs = -cp.log(probs[cp.arange(len(y_batch)), y_batch] + 1e-7)
            loss += cp.sum(log_probs)

            predictions = cp.argmax(probs, axis=1)
            num_correct += cp.sum(predictions == cp.asarray(y_batch))

            # backward pass
            gradient = probs.copy()
            gradient[cp.arange(len(y_batch)), y_batch] -= 1

            # normalize gradient by batch size
            gradient /= len(y_batch)

            # backpropagate
            gradient = softmax.backward(gradient, lr)
            
            gradient = pool4.backward(gradient)
            gradient = relu4.backward(gradient)
            gradient = conv4.backward(gradient, lr)
            
            gradient = pool3.backward(gradient)
            gradient = relu3.backward(gradient)
            gradient = conv3.backward(gradient, lr)
            
            gradient = pool2.backward(gradient)
            gradient = relu2.backward(gradient)
            gradient = conv2.backward(gradient, lr)
            
            gradient = pool1.backward(gradient)
            gradient = relu1.backward(gradient)
            gradient = conv1.backward(gradient, lr)

            # calculate stats by images processed so far
            images_processed = i + len(x_batch)
            avg_loss = loss / images_processed
            avg_acc = num_correct / images_processed

        print(f"End of Epoch {epoch+1}: Loss {avg_loss:.4f} | Accuracy: {avg_acc:.2%}")

        # learning rate decay
        lr = lr * 0.98
    
    # return trained layers to save them
    return [conv1, conv2, conv3, conv4], [relu1, relu2, relu3, relu4], [pool1, pool2, pool3, pool4], softmax

def normalize_feature_map(fm):
    # normalize single feature map to [0,1] for visualization
    fm = fm - fm.min()
    if fm.max() > 0:
        fm = fm / fm.max()
    return fm

def select_top_filters(feature_maps, top_k=6):
    # select features w hightest mean activation

    # ensure on cpu
    if isinstance(feature_maps, cp.ndarray):
        feature_maps = cp.asarray(feature_maps)

    # compute per channel
    means = feature_maps.mean(axis=(1, 2))
    top_indices = np.argsort(means)[-top_k:][::-1]
    return top_indices


def save_feature_grid(feature_maps, layer_name, save_dir):
    # move to cpu
    if isinstance(feature_maps, cp.ndarray):
        feature_maps = cp.asnumpy(feature_maps)

    os.makedirs(save_dir, exist_ok=True)

    # combine features into heatmap
    single_map = feature_maps.mean(axis=0)
    
    # normalize to [0,1]
    single_map = single_map - single_map.min()
    if single_map.max() > 0:
        single_map = single_map / single_map.max()

    # setup figure
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # add smoothing
    ax.imshow(single_map, cmap='viridis', interpolation='bicubic')

    # save
    plt.savefig(os.path.join(save_dir, f"{layer_name}.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def visualize_activations(image_array, convs, relus, pools, save_dir="docs/assets"):
    # run forward pass on one image and save activations

    # convert to gpu
    x = cp.asarray(image_array[np.newaxis, :], dtype=cp.float32)

    # block 1
    out = convs[0].forward(x)
    out = relus[0].forward(out)
    out = pools[0].forward(out)
    save_feature_grid(cp.asnumpy(out[0]), "layer1", save_dir)

    # block 2
    out = convs[1].forward(out)
    out = relus[1].forward(out)
    out = pools[1].forward(out)
    save_feature_grid(cp.asnumpy(out[0]), "layer2", save_dir)

    # block 3
    out = convs[2].forward(out)
    out = relus[2].forward(out)
    out = pools[2].forward(out)
    save_feature_grid(cp.asnumpy(out[0]), "layer3", save_dir)

    # block 4
    out = convs[3].forward(out)
    out = relus[3].forward(out)
    out = pools[3].forward(out)
    save_feature_grid(cp.asnumpy(out[0]), "layer4", save_dir)

    print("Activation visualizations saved.")


if __name__ == "__main__":
    # configuration
    DATA_DIR = "data"

    # load data
    x_train, y_train, class_names = load_data(DATA_DIR)
    num_classes = len(class_names)

    print(f"Loaded {len(x_train)} images.")

    if len(x_train) == 0:
        print("WARN: No images found.")
    else:
        # check if data needs normalization
        if x_train.max() > 1.0:
            x_train = x_train / 255.0

        EPOCHS = 100
        BATCH_SIZE = 64

        print(f"Starting training on GPU ({EPOCHS} epochs)...")
        print("-" * 50)

        start_time = time.time()
        
        # run training
        convs, relus, pools, sm = train(x_train, y_train, num_classes=num_classes, num_epochs=EPOCHS, lr=0.005)

        end_time = time.time()

        # stat calculation
        total_time = end_time - start_time
        total_images_processed = len(x_train) * EPOCHS
        images_per_second = total_images_processed / total_time

        print("\n" + "="*40)
        print("       TRAINING PERFORMANCE REPORT       ")
        print("="*40)
        print(f"Device:            {get_gpu_name()}")
        print(F"Images:            {len(x_train)}")
        print(f"Batch Size:        {BATCH_SIZE}")
        print(f"Total Operations:  {total_images_processed} images processed")
        print("-" * 40)
        print(f"Total Time:        {total_time:.2f} seconds")
        print(f"Throughput:        {images_per_second:.0f} images/sec")
        print("="*40)

        print("Generating Visualizations...")

        # use first training image
        sample_image = x_train[0]

        visualize_activations(sample_image, convs, relus, pools)

        print("Exporting model...")

        export_to_json(convs, sm)
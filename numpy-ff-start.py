import numpy as np
import time

"""Pure numpy feed forward neural network I did for practice"""


NUM_SAMPLES = 2048
INPT_SIZE = 50
OUT_SIZE = 10
HIDDEN_LAYERS = [100]
ETA = 1e-5
EPOCHS = 300

def rule(inpt, resize):
    inpt = np.matmul(inpt, resize)
    try:
        inpt_highest = np.argmax(inpt, axis=1)
    except IndexError:
        inpt_highest = np.argmax(inpt)
    inpt = np.zeros_like(inpt)
    try:
        inpt[np.arange(inpt.shape[0]), inpt_highest] = 1.
    except IndexError:
        inpt[inpt_highest] = 1.
    return inpt


def forward(weights, ans):
    assert (len(weights) >= 1)
    acts = [ans]
    zs = [ans]
    for weight in weights[:-1]:
        ans = ans.dot(weight)
        zs.append(ans.copy())
        ans = relu(ans)
        acts.append(ans.copy())
    ans = ans.dot(weights[-1])
    zs.append(ans.copy())
    ans = softmax(ans)
    return acts, zs, ans


def identity(inpt):
    return inpt

def identity_prime(inpt):
    return 1

def tanh(inpt):
    return np.tanh(inpt)

def tanh_prime(inpt):
    tanh_inpt = tanh(inpt)
    return 1. - tanh_inpt ** 2

def sigmoid(inpt):
    return 1. / (1. + np.exp(-inpt))

def sigmoid_prime(inpt):
    sigm = sigmoid(inpt)
    return sigm * (1. - sigm)

def softmax(inpt):
    try: 
        e_x = np.exp(inpt - np.expand_dims(np.max(inpt, axis=1), axis=1))
    except IndexError:
        e_x = np.exp(inpt - np.max(inpt))
        return e_x / e_x.sum()
    return e_x / np.expand_dims(e_x.sum(axis=1), axis=1)


def softmax_prime(acts, answers):
    # this is actually dL/dz, not just da/dz
    return acts - answers


def relu(inpt):
    return np.maximum(inpt, 0)

#def relu_prime(inpt):
#    # Useless and slow function, can combine multiplication and this func into one easily
#    inpt = inpt.clip(min=0)
#    inpt[inpt > 0] = 1.
#    return inpt

def get_grads(activations, weighted_zs, weights, grad_init, answers=None, y_hat=None):
    # error = grad_init * sigmoid_prime(weighted_zs[-1])  # the 1 is activation_func_prime(z_(L-1)), so this term is dL/dz 
    error = softmax_prime(y_hat, answers)    # for softmax, this is dL/dz in one term
    grads = []
    for act, z, weight in zip(activations[::-1], weighted_zs[-2::-1], weights[::-1]):
        grads.append(act.T.dot(error))
        error = error.dot(weight.T) # * sigmoid_prime(z)
        # error[z < 0] = 0     # combines relu_prime(z) and multiplication with error 
    return grads[::-1]
   

def get_loss(y_predicted, y_actual):
    return -np.sum(y_actual * np.log(y_predicted))


def grad_check(bkprp_grads, weights, x_in, y_out):
    h = 1e-4
    accept = 1e-3
    for layer_num, (layer, grad) in enumerate(zip(weights, bkprp_grads)):
        it = np.nditer(layer, ['multi_index'], ['readwrite'])
        while not it.finished:
            ix = it.multi_index
            layer[ix] += h
            loss_plus = get_loss(forward(weights, x_in)[2], y_out)
            layer[ix] -= 2. * h
            loss_minus = get_loss(forward(weights, x_in)[2], y_out)
            layer[ix] += h
            grad_comp = (loss_plus - loss_minus) / (2. * h)
            diff = abs(grad_comp - grad[ix]) / max(1, abs(grad_comp), abs(grad[ix]))
            if diff > accept:
                print(f"Grad check failed at layer {layer_num}, index {ix}")
                print(f"Computed gradient {grad[ix]}, actual gradient {grad_comp}, difference was {diff}")
                quit()
            it.iternext()
        print(f"Gradcheck for layer {layer_num} passed")
    print("Gradient check passed")
        
def test(inpt_x, trained_weights, rule_weights):
    result = forward(trained_weights, inpt_x)[2]
    actual = rule(inpt_x, rule_weights)
    print("network", result)
    print("actual", actual)
    print("net answer", np.argmax(result))
    print("real answer", np.argmax(actual))
    print("loss", get_loss(result, actual))

def apply_grads(weights, gradients):
    for idx, (weight, gradient) in enumerate(zip(weights, gradients)):
        weights[idx] =  np.clip(weight - ETA * gradient, -50., 50.)

def main():
    try:
        w = [np.random.randn(INPT_SIZE, HIDDEN_LAYERS[0])]
        try:
            w += [np.random.randn(h0, h1) for h0, h1 in zip(HIDDEN_LAYERS[:-1], HIDDEN_LAYERS[1:])]
        except IndexError:
            pass
        w += [np.random.randn(HIDDEN_LAYERS[-1], OUT_SIZE)]
    except IndexError:
        w = [np.random.randn(INPT_SIZE, OUT_SIZE)]

    x = np.random.randn(NUM_SAMPLES, INPT_SIZE) * 5.0
    resizer = np.random.uniform(0, 1, size=(INPT_SIZE, OUT_SIZE))
    y = rule(x, resizer)
    x_test = np.random.randn(100, INPT_SIZE)
    y_test = rule(x_test, resizer)
    start = time.time()
    for t in range(EPOCHS):
        activations, pre_activations, y_hat = forward(w, x)
        loss = np.square(y - y_hat).sum()
        #print(t, loss)
        grad_y_hat = 2.0 * (y_hat - y)
        nabla_ws = get_grads(activations, pre_activations, w, grad_y_hat, y, y_hat)
        #print("grads ", nabla_ws)
        #[print(g.shape) for g in nabla_ws]
        #print("#"*100)
        #print("w init ", w)
        #print("#"*100)
        #grad_check(nabla_ws, w, x, y)
        apply_grads(w, nabla_ws)
        test_answers = forward(w, x_test)[2]
        test_loss = np.square(y_test - test_answers).sum()
        print(t, test_loss) 
        #print("w final ", w)
        #[print(np.amax(the_weight)) for the_weight in w]
    test_value = np.random.randn(INPT_SIZE)
    test(test_value, w, resizer)
    print("TOTAL TIME FOR TRAINING:", time.time() - start)
if __name__ == "__main__":
    main()



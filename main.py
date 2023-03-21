import numpy as np
import PIL.Image as im

"""def gradient(z,f): # gradient = [1,i]*diff
    epsilon = 0.01
    kernel = epsilon * (
            np.outer(np.linspace(-1, 1, 3), np.full(3, 1)) + 1j * np.outer(np.full(3, 1), np.linspace(-1, 1, 3)))
    f_vec = np.vectorize(f)
    space = z+kernel
    grad = np.gradient(f_vec(space))
    return np.array(list(map(lambda a:a[1][1]/epsilon,grad)))"""

def diff(z,f):
    epsilon = 0.01
    kernel = epsilon*np.linspace(-1, 1, 3)
    f_vec = np.vectorize(f)
    space = z+kernel
    grad = np.gradient(f_vec(space))
    return grad[1]/epsilon

def sym_diff(z,f):
    if z == 0:
        return 1
    if f == my_poly:
        return 3*z**2
    else:
        print("wrong func")

def my_poly(x):
    return x**3

def attractors(a):
    return

def classify(z, roots=[1,-0.5+0.8660254j,-0.5-0.8660254j]):
    epsilon = 0.001
    if abs(z-roots[0])<epsilon:
        return 1
    elif abs(z-roots[1])<epsilon:
        return 2
    elif abs(z-roots[2])<epsilon:
        return 3
    else:
        return 0

def colorize(z, roots=[1,-0.5+0.8660254j,-0.5-0.8660254j]):
    epsilon = 0.001
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255)]
    for c, r in zip(colors, roots):
        if abs(z-r)<epsilon:
            return c
    return (0,0,0)

def search(l, min, max):
    diff = max-min
    avg = (max+min)/2
    print(min, max, len(l))
    if abs(diff) < 0.1**10:
        yield avg
    else:
        imquads = [[i for i in l if np.imag(i-avg) < 0 and np.real(i-avg) < 0],[i for i in l if np.imag(i-avg) >= 0 and np.real(i-avg) < 0],[i for i in l if np.imag(i-avg) < 0 and np.real(i-avg) >= 0],[i for i in l if np.imag(i-avg) >= 0 and np.real(i-avg) >= 0]]
        for r in range(4):
            if len(imquads[r]) > len(l)/16:
                shift = (np.real(diff)/2 if r==2 or r==3 else 0) + (1j*np.imag(diff)/2 if r==1 or r==3 else 0)
                for i in search(imquads[r], min + shift, avg + shift):
                    yield i


def newtons_method():
    size = 1001

    view_field = np.outer(np.linspace(-1,1,size),np.full(size,1))+1j*np.outer(np.full(size,1),np.linspace(-1,1,size))
    #print(view_field)
    for i in range(20):
        view_field = view_field - np.vectorize(lambda z:my_poly(z)/diff(z,my_poly))(view_field)

    img = im.new('RGB', (size, size))

    px = img.load()

    pixels = np.vectorize(classify)(view_field)

    for i in range(size):
        for j in range(size):
            px[i,j] = (255,0,0) if pixels[i][j] == 1 else ((0,255,0) if pixels[i][j] == 2 else ((0,0,255) if pixels[i][j] == 3 else (0,0,0)))

    img.save('image.png')

def sym_newtons_method():
    size = 1001

    view_field = np.outer(np.linspace(-1,1,size),np.full(size,1))+1j*np.outer(np.full(size,1),np.linspace(-1,1,size))
    #print(view_field)
    for i in range(30):
        print(i)
        view_field = view_field - np.vectorize(lambda z:my_poly(z)/sym_diff(z,my_poly))(view_field)

    #print([i for i in np.asarray(np.unique(np.ravel(view_field), return_counts=True))[1]])

    #print([n for n, i in np.asarray(np.unique(np.ravel(view_field), return_counts=True)).T if np.real(i) > 10])

    roots = [i for i in search(np.ravel(view_field), -10.1-10.1j, 10+10j)]
    #roots = [i for i in search([n for n, i in np.asarray(np.unique(np.ravel(view_field), return_counts=True)).T if np.real(i) > 10], -10.1-10.1j, 10+10j)]
    print(roots)

    #print([i for i in search(np.ravel(view_field), -10-10j, 10+10j)])

    #pixels = np.transpose(np.array(np.vectorize(colorize)(view_field)), axes=(2,1,0)).astype("uint8")

    pixels = np.transpose(np.array(np.vectorize(lambda x:colorize(x,roots = roots))(view_field)), axes=(2,1,0)).astype("uint8")

    """print("pixels",pixels)

    test_img = im.open("image.png")

    print("test image",np.asarray(test_img))

    print(pixels.dtype, np.asarray(test_img).dtype)"""

    img = im.fromarray(pixels, mode="RGB")

    img.save('image2.png')

if __name__ == '__main__':
    sym_newtons_method()
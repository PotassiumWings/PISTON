import matplotlib.pyplot as plt


def get_x(f):
    res = []
    x = []
    for line in f.readlines():
        if not "train loss" in line:
            continue
        val = line.split(" ")[13][:-1]
        res.append(float(val))
        x.append(len(x))
    return x[200:], res[200:]


f = open("1.txt", "r")  # MTGNN
x1, y1 = get_x(f)
plt.plot(x1, y1, label="MTGNN")

f = open("3.txt", "r")  # STGCN
x1, y1 = get_x(f)
plt.plot(x1, y1, label="STGCN")

f = open("4.txt", "r")  # STSSL
x1, y1 = get_x(f)
plt.plot(x1, y1, label="STSSL")

f = open("2.txt", "r")
x1, y1 = get_x(f)
plt.plot(x1, y1, label="Running")

plt.legend()
plt.show()

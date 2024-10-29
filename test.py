import torch
import matplotlib.pyplot as plt
def main():
    print(torch.__version__)
    print(torch.__path__)
    # print(torch.nn.Conv2d.__doc__)

    m = torch.nn.ReLU()
    input = torch.randn(10) # generate random tensor
    print(input)
    output = m(input)
    print(output)
    # print(torch.randn.__doc__)
    # print(plt.axhline.__doc__)

    # 数据
    x = [1, 2, 3, 4, 5]
    y1 = [1, 4, 9, 16, 25]
    y2 = [1, 2, 3, 4, 5]

    # 绘制两条线并添加标签
    plt.plot(x, y1, label='Square Numbers')
    plt.plot(x, y2, label='Linear Numbers')

    # 显示图例
    plt.legend()

    # 添加标题和标签
    plt.title('Example Plot')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    # 显示图表
    plt.show()


if __name__ == "__main__":
    main()
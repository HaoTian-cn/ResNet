from Model import *

def train_model(net,train_loader,validate_loader):
    net=net.to(device)
    loss_function = nn.CrossEntropyLoss()
    loss_function=loss_function.to(device)
    # 抽取模型参数
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.NAdam(params, lr=0.0001,momentum_decay=0.5)
    # lf = lambda x: ((1 + math.cos(x * math.pi / 300)) / 2) * (1 - 0.0001) + 0.0001
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # 迭代次数（训练次数）
    epochs = 600
    # 用于判断最佳模型
    best_acc = 0.0
    # 最佳模型保存地址
    save_path = './resNet34.pth'
    train_steps = len(traindata_loader)

    #画图
    train_acc_list=[]
    val_acc_list=[]
    train_loss_list=[]

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        t_acc=0.0
        t_num=0
        # tqdm：进度条显示
        train_bar = tqdm(train_loader, file=sys.stdout,colour='blue')
        # train_bar: 传入数据（数据包括：训练数据和标签）
        # enumerate()：将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中
        # enumerate返回值有两个：一个是序号，一个是数据（包含训练数据和标签）
        # x：训练数据（inputs）(tensor类型的），y：标签（labels）(tensor类型）
        for step, data in enumerate(train_bar):
            # 前向传播
            images, labels = data
            t_num+=len(labels)
            # 计算训练值
            logits = net(images.to(device))
            # 计算训练准确率
            predict_y = torch.max(logits, dim=1)[1]
            t_acc += torch.eq(predict_y, labels.to(device)).sum().item()
            # 计算损失
            loss = loss_function(logits, labels.to(device))
            # 反向传播
            # 清空过往梯度
            optimizer.zero_grad()
            # 反向传播，计算当前梯度
            loss.backward()
            optimizer.step()

            # item()：得到元素张量的元素值
            running_loss += loss.item()

            # 进度条的前缀
            # .3f：表示浮点数的精度为3（小数位保留3位）
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # scheduler.step()
        tra_accurate = t_acc / t_num
#-------------------------------------------------------------------------------------------------------------------------
        # 测试
        # eval()：如果模型中有Batch Normalization和Dropout，则不启用，以防改变权值
        net.eval()
        acc = 0.0
        val_num=0
        # 清空历史梯度，与训练最大的区别是测试过程中取消了反向传播
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout,colour='green')
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_num+=len(val_labels)
                outputs = net(val_images.to(device))
                # torch.max(input, dim)函数
                # input是具体的tensor，dim是max函数索引的维度，0是每列的最大值，1是每行的最大值输出
                # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
                predict_y = torch.max(outputs, dim=1)[1]
                # 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False
                # .sum()对输入的tensor数据的某一维度求和
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f tra_accurate: %.3f val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, tra_accurate,val_accurate))

        # 保存最好的模型权重
        if val_accurate > best_acc:
            best_acc = val_accurate
            # torch.save(state, dir)保存模型等相关参数，dir表示保存文件的路径+保存文件名
            # model.state_dict()：返回的是一个OrderedDict，存储了网络结构的名字和对应的参数
            torch.save(net.state_dict(), save_path)

        #保存每轮训练精度
        train_acc_list.append(tra_accurate) #训练准确率
        val_acc_list.append(val_accurate) #验证准确率
        train_loss_list.append(running_loss) #损失
    print('Finished Training')
    #画图
    plt.figure()
    plt.plot(range(len(train_loss_list)),train_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.show()
    plt.figure()
    plt.plot(range(len(train_acc_list)),train_acc_list,label='train')
    plt.plot(range(len(val_acc_list)),val_acc_list,color='r',label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Accurate')
    plt.legend()
    plt.show()



def findbest(model,validate_loader):
    model.load_state_dict(torch.load('./resNet34.pth', map_location=device))
    model=model.to(device)
    model.eval()
    best_var=100
    with torch.no_grad():
        # 预测类别
        for img,_ in validate_loader:
            # squeeze()：维度压缩，返回一个tensor（张量），其中input中大小为1的所有维都已删除
            output = model(img.to(device))
            var = torch.std(output,dim=1)
            min_var_id=var.argmin()
            if var[min_var_id]<best_var:
                best_tensor=img[min_var_id]
                best_var=var[min_var_id]

    plt.imshow(best_tensor.squeeze(0))
    plt.show()
    image = Image.fromarray((np.array(best_tensor.squeeze(0)) * 255).astype(np.uint8))
    image.save('./best_img_40.bmp')

def predict(model,path):
    model.load_state_dict(torch.load('./best_model/resNet34_40.pth', map_location=device))
    model=model.to(device)
    model.eval()
    trans=  transforms.Compose([
            transforms.CenterCrop(40),
            transforms.ToTensor()          #把图片进行归一化，并把数据转换成Tensor类型
        ])
    img = trans(Image.open(path).convert('L'))

    with torch.no_grad():
        output = model(img.unsqueeze(0).to(device))
        print(np.array(output.cpu().view(-1)))
        id=output.argmax()
        print('该截图来自tuxing{}'.format(id+1))


if __name__ == '__main__':
    traindata_40=Dataset('./','train',40)
    testdata_40=Dataset('./','test',40)
    traindata_loader = DataLoader(dataset=traindata_40, batch_size=256, shuffle=True)
    testdata_loader = DataLoader(dataset=testdata_40, batch_size=256, shuffle=False)
    for i, data in enumerate(traindata_loader):
        images, labels = data

        # 打印数据集中的图片
        img = torchvision.utils.make_grid(images).numpy()
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.show()
        print(images.shape)
        print(labels)

        break
    print(len(testdata_loader))
    # train_model(resnet34(60),traindata_loader,testdata_loader)
    # findbest(resnet34(60),traindata_loader)
    predict(resnet34(60),r'T:\模拟练习1\2023第一次模拟题(发布)\2023第一次模拟题(发布)\A1题：最强大脑之图形辨识\附件\附件\截图1.png')
    # predict(resnet34(60), r'T:\模拟练习1\2023第一次模拟题(发布)\2023第一次模拟题(发布)\A1题：最强大脑之图形辨识\附件\附件\截图2.png')
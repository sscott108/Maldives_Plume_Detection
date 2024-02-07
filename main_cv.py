#!/usr/bin/env python
# coding: utf-8

# In[1]:


from thil_utils import *


# In[ ]:





# In[ ]:


p = create_dataset('kappa_data.pkl',
                   apply_transforms=False)
n = create_dataset_n(img_dir = '/hpc/home/srs108/thilafushi/none_sorted_images/negative',
                    apply_transforms=False)


# In[17]:


def train(model, loss, opt, epoch, dataloader, device):
    model.train()
    train_ious = []
    running_loss = 0
    running_acc = 0

    for i, batch in enumerate(dataloader):
        x = batch['img'].float()
        y = batch['fpt'].unsqueeze(dim=1)   
        output = model(x)

        #Accuracy
        acc_epoch = pixel_accuracy(output, y)

        #Binary Cross Entropy Loss
        running_acc += acc_epoch.item()
        
        #Binary Cross Entropy Loss
        loss_epoch = loss(output, y.float())
        running_loss += loss_epoch.item()

        #IoU
        output_binary = np.zeros(output.shape)
        output_binary[output.cpu().detach().numpy() >= 0.5] = 1

        for j in range(y.shape[0]):
            z = jaccard_score(y[j].flatten().cpu().detach().numpy(),
                      output_binary[j][0].flatten())
            if (np.sum(output_binary[j][0]) != 0 and np.sum(y[j].cpu().detach().numpy()) != 0):
                train_ious.append(z)     
        opt.zero_grad()
        loss_epoch.backward()
        opt.step()

    return running_loss /len(dataloader), np.average(train_ious),running_acc/len(dataloader)


# In[33]:


def test(model, loss, scheduler, epoch, dataloader, device, val_o_test = 'val'):
    model.eval()
    running_loss = 0
    running_acc = 0
    test_ious = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x = batch['img'].float()
            y = batch['fpt'].unsqueeze(dim=1)
            output = model(x)
        
            #Accuracy
            acc_epoch = pixel_accuracy(output, y)
            running_acc += acc_epoch.item()

            #Binary Cross Entropy Loss
            loss_epoch = loss(output, y.float())
            running_loss += loss_epoch.item()

            output_binary = np.zeros(output.shape)
            output_binary[output.cpu().detach().numpy() >= 0.5] = 1
            
            #IoU
            for k in range(y.shape[0]):
                z = jaccard_score(y[k].flatten().cpu().detach().numpy(),
                          output_binary[k][0].flatten())
                if (np.sum(output_binary[k][0]) != 0 and
                    np.sum(y[k].cpu().detach().numpy()) != 0):
                    test_ious.append(z)
            
            torch.cuda.empty_cache()

            if val_o_test =='test':
                for n in range(16):
                    try:
                        image_comparison(batch['img'][n], y[n][0].cpu(), output_binary[n][0],
                                save=True, show=False, train_o_test='Testing', 
                                fig_name='Test_{}_{}'.format(batch['imgfile'][n].split('/')[-2],
                                batch['imgfile'][n].split('/')[-1].split('.')[0].split('_')[0]))

                    except Exception as e:
                        print(e)
                        continue

        return running_loss/len(dataloader), np.average(test_ious), running_acc/len(dataloader)


# In[37]:


epochs = 200
loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(14))
best_iou = 0.0 
best_model_weights = None 
patience = 30
history = {'epoch':[],'train_loss': [], 'val_loss':[], 
           'train_iou': [], 'val_iou':[], 'train_acc': [], 'val_acc':[]}

#split data in testing, training, validationing
data_train =  torch.utils.data.ConcatDataset([p, n])
train_data, test_data = train_test_split(data_train, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

model = smp.Unet(encoder_name = 'resnet18', encoder_weights = 'imagenet', in_channels = 3,classes = 1, activation='sigmoid')
opt  = optim.SGD(model.parameters(), lr=0.0005920815839322062, momentum=0.9880160542000381)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, threshold=0.0005920815839322062, min_lr=1e-6)

train_dl = DataLoader(train_data, batch_size=32,shuffle=True)
val_dl = DataLoader(val_data, batch_size = 16, shuffle=False, drop_last=True)
test_dl = DataLoader(test_data, batch_size = 16, shuffle=False)

for epoch in range(1, epochs+1):
    train_loss, train_iou, train_acc = train(model, loss, opt, epoch, train_dl, device)
    val_loss, val_iou, val_acc = test(model, loss, scheduler, epoch, val_dl, device, val_o_test='val')
    print(f"Current Train and Val IoU on epoch {epoch} : {train_iou}, {val_iou}")
    
    if val_iou > best_iou:
        trigger_times = 0
        best_model_weights = model.state_dict()
        torch.save(model.state_dict(), 'best_model_weights.pt')
        
    else: 
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping on epoch {epoch} - patience reached")
            break

    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['train_iou'].append(train_iou)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_iou'].append(val_iou)
    history['val_acc'].append(val_acc)


# In[38]:


# Load the best model weights and test the model
model.load_state_dict(torch.load('best_model_weights.pt'))
test_loss, test_iou, test_acc = test(model, loss, scheduler, 0, test_dl, device, val_o_test='test')
print('Best Testing IoU:',test_iou)
print('Best Testing Accuracy',test_acc)


# In[39]:


# import time
# start_time = time.time()
# main(epochs=2, device=device) #history,posdf = 
# print(f"Run time: {( time.time() - start_time)/3600}h")
df = pd.DataFrame(history)


# # EDIT BEFORE RUNNING

# In[40]:


print('Best Train IoU:',max(df['train_iou']))
print('Best Val IoU', max(df['val_iou']))
print('Best Train Accuracy:', max(df['train_acc']))
print('Best Test Accuracy:', max(df['val_acc']))


# In[10]:


train_test_loss(df['train_loss'], df['val_loss'], len(df['val_loss']), save = True, fig_name='loss')


# In[11]:


train_test_ious(df['train_iou'], df['val_iou'], len(df['val_iou']), save=True, fig_name='iou')


# In[11]:


train_test_acc(df['train_acc'], df['val_acc'], len(df['val_acc']), save = True, fig_name='acc')


# In[13]:


df.to_csv('history.csv', index=False)


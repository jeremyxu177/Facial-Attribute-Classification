  

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from dataloader import *
from model import *
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

#%%
train_index = list(range(40000))# set index of training set 
valid_index = list(range(41000,42000))  # set index of validation set 
Dtrain = SubsetRandomSampler(train_index) # get 19000 trainset
Dvil = SubsetRandomSampler(valid_index) # get 1000 validation set

#epoch =  5
batch_size = 50
train_data = loaddataset(path='data/',
                                transforms=transforms.Compose([
                                    transforms.Resize((178,218)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
if torch.cuda.is_available(): # judge whether to use gpu
    num_worker={}
else:
    num_worker = {'num_workers': 0}
    
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, sampler=Dtrain, **num_worker
    )

valid_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size,  sampler=Dvil,**num_worker
    )


model = AttrPre()
if torch.cuda.is_available(): # judge whether to use gpu
    model.cuda()

'''Set up MSE loss'''
criterion = nn.MSELoss()

#%%
    
    
def train(epoch, model, loss_fn, optimizer, dataloader,log_interval=50):
    model.train()
    train_loss = 0
    best_loss = 10
    count = 0
    for i_batch, sample_batched in enumerate(dataloader):
        optimizer.zero_grad()
        img, Attractive, EyeGlasses, Male, MouthOpen, Smiling, Young, brown,hat,lip,oval = Variable(sample_batched['image'].cuda()), \
                Variable(sample_batched['Attractive'], requires_grad=False),\
                Variable(sample_batched['EyeGlasses'], requires_grad=False), \
                Variable(sample_batched['Male'], requires_grad=False), \
                Variable(sample_batched['MouthOpen'], requires_grad=False), \
                Variable(sample_batched['Smiling'], requires_grad=False), \
                Variable(sample_batched['Young'], requires_grad=False),\
                Variable(sample_batched['brown hair'], requires_grad=False),\
                Variable(sample_batched['hat'], requires_grad=False),\
                Variable(sample_batched['lipstick'], requires_grad=False),\
                Variable(sample_batched['oval face'], requires_grad=False)
        AttractiveF = Attractive.type((torch.FloatTensor))
        EyeGlassesF = EyeGlasses.type((torch.FloatTensor))
        MaleF = Male.type((torch.FloatTensor))
        MouthOpenF = MouthOpen.type((torch.FloatTensor))
        SmilingF = Smiling.type((torch.FloatTensor))
        YoungF = Young.type((torch.FloatTensor))
        brownF = brown.type((torch.FloatTensor))
        hatF = hat.type((torch.FloatTensor))
        lipF = lip.type((torch.FloatTensor))
        ovalF = oval.type((torch.FloatTensor))
        AttractivePre, EyeGlassesPre, MalePre, MouthOpenPre, SmilingPre, YoungPre,brownPre,hatPre,lipPre,ovalPre = model(img)
        lossAttractive = loss_fn(AttractivePre,AttractiveF.cuda())
        lossEyeGlasses = loss_fn(EyeGlassesPre, EyeGlassesF.cuda())
        lossMale = loss_fn(MalePre, MaleF.cuda())
        lossMouthOpen = loss_fn(MouthOpenPre, MouthOpenF.cuda())
        lossSmiling = loss_fn(SmilingPre, SmilingF.cuda())
        lossYoung = loss_fn(YoungPre, YoungF.cuda())
        lossbrown = loss_fn(brownPre, brownF.cuda())
        losshat = loss_fn(hatPre, hatF.cuda())
        losslip = loss_fn(lipPre, lipF.cuda())
        lossoval = loss_fn(ovalPre, ovalF.cuda())
        '''
        set special weight for different attributes
        '''
        total1 = lossAttractive + lossEyeGlasses + lossMale + lossMouthOpen + lossSmiling + lossYoung+lossbrown+losshat+lossoval+losslip
        w1 = lossAttractive/total1*10
        w2 = lossEyeGlasses/total1*10
        w3 = lossMale/total1*10
        w4 = lossMouthOpen/total1*10
        w5 = lossSmiling/total1*10
        w6 = lossYoung/total1*10
        w7 = lossbrown/total1*10
        w8 = losshat/total1*10
        w9 = losslip/total1*10
        w0 = lossoval/total1*10
        loss = w1*lossAttractive + w2*lossEyeGlasses + w3*lossMale + w4*lossMouthOpen + w5*lossSmiling + w6*lossYoung+w7*lossbrown+w8*losshat+w9*lossoval+w0*losslip

        loss.backward()

        optimizer.step()
        train_loss += loss.data.cpu().numpy()
        if i_batch % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, i_batch , len(dataloader),
                100. * i_batch / len(dataloader), loss.data))

            count +=1
            if count == 5:
                count = 0
#                print(lossAttractive)
#                print(lossEyeGlasses)
#                print(lossMale)
#                print(lossMouthOpen)
#                print(lossSmiling)
#                print(lossYoung)
#                print(lossbrown)
#                print(losshat)
#                print(losslip)
#                print(lossoval)
#                print(w1)
#                print(w2)
#                print(w3)
#                print(w4)
#                print(w5)
#                print(w6)
#                print(w7)
#                print(w8)
#                print(w9)
#                print(w0)
                if loss.data <best_loss:
                    ### save the model 
                    best_loss =  loss.data 
                    torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
    
                            }, 'M:/Data_Baidu/CelebA/Img/img_align_celeba_png.7z/img_align_celeba_png.7z/params.pkl')
                    print('new best model saved at epoch: {}'.format(epoch))
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
#    add(line)
    return train_loss

'''Set up Adam optimizer, with 1e-3 learning rate and betas=(0.9, 0.999). '''
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
  
    
#%%

def test(model, loss_fn, dataloader,log_interval=50):

    test_loss = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
    
            img, Attractive, EyeGlasses, Male, MouthOpen, Smiling, Young, brown,hat,lip,oval = Variable(sample_batched['image'].cuda()), \
                Variable(sample_batched['Attractive'], requires_grad=False),\
                Variable(sample_batched['EyeGlasses'], requires_grad=False), \
                Variable(sample_batched['Male'], requires_grad=False), \
                Variable(sample_batched['MouthOpen'], requires_grad=False), \
                Variable(sample_batched['Smiling'], requires_grad=False), \
                Variable(sample_batched['Young'], requires_grad=False),\
                Variable(sample_batched['brown hair'], requires_grad=False),\
                Variable(sample_batched['hat'], requires_grad=False),\
                Variable(sample_batched['lipstick'], requires_grad=False),\
                Variable(sample_batched['oval face'], requires_grad=False)
            AttractiveF = Attractive.type((torch.FloatTensor))
            EyeGlassesF = EyeGlasses.type((torch.FloatTensor))
            MaleF = Male.type((torch.FloatTensor))
            MouthOpenF = MouthOpen.type((torch.FloatTensor))
            SmilingF = Smiling.type((torch.FloatTensor))
            YoungF = Young.type((torch.FloatTensor))
            brownF = brown.type((torch.FloatTensor))
            hatF = hat.type((torch.FloatTensor))
            lipF = lip.type((torch.FloatTensor))
            ovalF = oval.type((torch.FloatTensor))   
            AttractivePre, EyeGlassesPre, MalePre, MouthOpenPre, SmilingPre, YoungPre,brownPre,hatPre,lipPre,ovalPre = model(img)
            lossAttractive = loss_fn(AttractivePre,AttractiveF.cuda())
            lossEyeGlasses = loss_fn(EyeGlassesPre, EyeGlassesF.cuda())
            lossMale = loss_fn(MalePre, MaleF.cuda())
            lossMouthOpen = loss_fn(MouthOpenPre, MouthOpenF.cuda())
            lossSmiling = loss_fn(SmilingPre, SmilingF.cuda())
            lossYoung = loss_fn(YoungPre, YoungF.cuda())
            lossbrown = loss_fn(brownPre, brownF.cuda())
            losshat = loss_fn(hatPre, hatF.cuda())
            losslip = loss_fn(lipPre, lipF.cuda())
            lossoval = loss_fn(ovalPre, ovalF.cuda())
            loss = lossAttractive + lossEyeGlasses + lossMale + lossMouthOpen + lossSmiling + lossYoung+lossbrown+losshat+lossoval+losslip
            
            test_loss = test_loss + loss
    test_loss /= len(dataloader)
    print('test set: Average loss: {:.4f}'.format(test_loss))
    return test_loss


#%%
for epoch in range(6):
    train_loss = train(epoch, model, criterion, optimizer, train_loader,  log_interval=50)
    test_loss = test(model, criterion, valid_loader) 
    


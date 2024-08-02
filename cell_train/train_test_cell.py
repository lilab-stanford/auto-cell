import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import timm
from small_CNN import Customresnet
from dataloaders import Cellclassification
from auto_cell.utils import *

def train_cell(args,cur,device):
    writer_dir = os.path.join(args.dataroot, 'run_cell', args.model_name+'_lr'+str(args.lr)+'_b'+str(args.batch_size)+'_opt'+args.opt+'_cur'+str(cur)+'_'+args.loss+'_match')
    if os.path.exists(writer_dir):
        shutil.rmtree(writer_dir,ignore_errors=True)
    os.makedirs(writer_dir,exist_ok=True)
    writer = SummaryWriter(writer_dir)

    print('\nInit train/val/test splits...', end=' ')
    train_imgs = np.load(os.path.join(args.cell_path, 'train_imgs.npy'))
    train_labels = np.load(os.path.join(args.cell_path, 'train_labels.npy'))
    val_imgs = np.load(os.path.join(args.cell_path, 'test_imgs.npy'))
    val_labels = np.load(os.path.join(args.cell_path, 'test_labels.npy'))
    target_imgs = np.load('target_imgs.npy')
    target_labels = np.load('target_labels.npy')
    print('Done!')
    print("Training on {} samples".format(len(train_labels)))
    print("Testing on {} samples".format(len(val_labels)))
    print("Target on {} samples".format(len(target_labels)))

    print('\nInit loss function...', end=' ')
    if args.loss=='ce':
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss=='focal':
        loss_fn = FocalLoss()
    loss_fn.to(args.gpu_ids[0])
    loss_fn_domain = nn.CrossEntropyLoss()
    loss_fn_domain.to(args.gpu_ids[0])
    print('Done!')

    print('\nInit Model...', end=' ')
    if args.model_name=='resnet18':
        model=Customresnet(num_classes=args.n_classes)
        pretrained_dict = torch.load(os.path.join(args.dataroot,'cell_dataset/byol-custom-dataset.ckpt'),map_location=torch.device('cpu'))['state_dict']
        for k in list(pretrained_dict.keys()):
            if "backbone" in k:
                pretrained_dict[k.replace("backbone.", "")] = pretrained_dict[k]
            del pretrained_dict[k]
        model.load_state_dict(pretrained_dict,strict=False)
    elif args.model_name=='resnet50':
        model = timm.create_model('resnet50', pretrained=True, num_classes=args.n_classes)
    elif args.model_name == 'VGG16':
        model = timm.create_model('vgg16', pretrained=True, num_classes=args.n_classes)
    elif args.model_name=='convnext':
        model = timm.create_model('convnext_atto', pretrained=True, num_classes=args.n_classes)
    elif args.model_name=='efficientnet':
        model = timm.create_model('efficientnet_lite0', pretrained=True, num_classes=args.n_classes)

    model.to(args.gpu_ids[0])
    model = torch.nn.DataParallel(model, args.gpu_ids)
    print('Done!')
    print_network(model)

    print("Params to learn:")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    train_data=Cellclassification(train_imgs,train_labels, split='train')
    val_data = Cellclassification(val_imgs,val_labels, split='val')
    target_data = Cellclassification(target_imgs,target_labels, split='target')
    kwargs = {'num_workers': 8, 'pin_memory': True}
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,shuffle=True,**kwargs)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, **kwargs)
    target_loader = DataLoader(dataset=target_data, batch_size=args.batch_size, shuffle=False, **kwargs)
    for epoch in tqdm(range(args.max_epoch)):
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn,loss_fn_domain,target_loader,args.da_s,args.da_t)
        validate(epoch, model, val_loader, args.n_classes, writer, loss_fn,'val')
        torch.save(model.state_dict(), os.path.join(args.dataroot, 'checkpoints/cell_cls', args.model_name+'_lr'+str(args.lr)+'_b'+str(args.batch_size)+'_opt'+args.opt+'_epoch'+str(epoch))+'_'+args.loss+'_{}.pt'.format(str(cur)))
    return
def train_loop(epoch, model, loader, optimizer, n_classes, writer, loss_fn,loss_fn_domain,target_loader,da_s,da_t):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.
    len_dataloader = min(len(loader), len(target_loader))
    data_target_iter = iter(target_loader)
    labels = np.zeros(len(loader.batch_sampler.sampler))
    preds = np.zeros(len(loader.batch_sampler.sampler))
    prob_idx=0
    for batch_idx, (data, label) in enumerate(loader):
        p = float(batch_idx + epoch * len_dataloader) / 10 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        data, label = data.to(device), label.to(device)
        batch_size = len(label)
        domain_label = torch.zeros(batch_size).long()
        domain_label = domain_label.to(device)
        logits,domain_output= model(data,alpha)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        err_s_label = loss_fn(logits, label)
        loss_value = err_s_label.item()
        err_s_domain = loss_fn_domain(domain_output, domain_label)
        data_target = data_target_iter.next()
        t_img, _ = data_target
        batch_size = len(t_img)
        domain_label = torch.ones(batch_size).long()
        t_img=t_img.to(device)
        domain_label=domain_label.to(device)
        _,domain_output= model(t_img,alpha)
        err_t_domain = loss_fn_domain(domain_output, domain_label)
        loss=err_s_label+da_s*err_s_domain+da_t*err_t_domain
        train_loss += err_s_label.item()+da_s*err_s_domain.item()+da_t*err_t_domain.item()
        prob_idx_add = len(label)
        labels[prob_idx:(prob_idx + prob_idx_add)] = label.cpu().numpy()
        preds[prob_idx:(prob_idx + prob_idx_add)] = Y_hat.squeeze().cpu().numpy()
        prob_idx = prob_idx + prob_idx_add
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(loader)
    train_error = calculate_error(preds, labels)
    print(' ')
    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

def validate(epoch, model, loader, n_classes, writer, loss_fn,split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.
    prob = np.zeros((len(loader.batch_sampler.sampler), n_classes))
    labels = np.zeros(len(loader.batch_sampler.sampler))
    preds = np.zeros(len(loader.batch_sampler.sampler))
    prob_idx=0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits, _= model(data,0)
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            Y_prob = F.softmax(logits, dim=1)
            loss = loss_fn(logits, label)
            prob_idx_add=len(label)
            prob[prob_idx:(prob_idx+prob_idx_add),:] = Y_prob.cpu().numpy()
            labels[prob_idx:(prob_idx+prob_idx_add)] = label.cpu().numpy()
            preds[prob_idx:(prob_idx + prob_idx_add)] = Y_hat.squeeze().cpu().numpy()
            prob_idx=prob_idx+prob_idx_add
            val_loss += loss.item()

    val_error =calculate_error(preds, labels)
    val_loss /= len(loader)
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        if split=='val':
            auc = roc_auc_score(labels, prob, multi_class='ovr')
        else:
            auc=roc_auc_score(labels, prob, multi_class='ovr')
    if writer:
        writer.add_scalar(split+'/loss', val_loss, epoch)
        writer.add_scalar(split+'/auc', auc, epoch)
        writer.add_scalar(split+'/error', val_error, epoch)
    print('\n{} Set, {}_loss: {:.4f}, {}_error: {:.4f}, auc: {:.4f}'.format(split,split,val_loss,split, val_error, auc))
    return

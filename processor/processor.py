import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import utils



def do_train(cfg,
             model,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             loss_fn,
             InfoNCE,
             num_query, local_rank,
             memory,
             memory2
             ):

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD  # epoch to save model
    eval_period = cfg.SOLVER.EVAL_PERIOD  # epoch to evaluate model

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    model = nn.DataParallel(model).cuda()
    
        

    loss_meter = AverageMeter()
    loss_momentum_contrast = AverageMeter()
    loss_info_nce = AverageMeter()
    acc_meter = AverageMeter()

    evaluator1 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator2 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    cross_entropy_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    # train

    for epoch in range(1, epochs + 1):
    
        start_time = time.time()
        loss_meter.reset()
        loss_momentum_contrast.reset()
        loss_info_nce.reset()
        acc_meter.reset()
        evaluator1.reset()
        evaluator2.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
    
            img = img.cuda()

            target = vid.cuda()
            target_cam = target_cam.cuda()
            target_view = target_view.cuda()
            # with amp.autocast(enabled=True):

            score, feat1_bn, feat2_bn,  feat1_recons_bn, feat2_recons_bn, x1_mask, x2_mask, mask= model(img, labels=target, memory_features=memory.features)
            
            momentum_contrast = memory(feat1_bn, target) + 0.3 * memory(feat2_bn, target, self_momentum=1)
            loss = mse_loss(x1_mask, torch.ones_like(x1_mask, requires_grad=False)) + mse_loss(x2_mask, mask) + loss_fn(score[0:2], None, target) + loss_fn(score[2:4], None, target) + loss_fn(score[4:6], None, target)

            info_nce = 0.5 * (memory2(feat1_recons_bn, target) + memory2(feat2_recons_bn, target, self_momentum=1))
           
            (loss + momentum_contrast + info_nce).backward()
            optimizer.step()

            if isinstance(score, list): # if there are multi classification scores, use the score from global features.
                acc = (score[1].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()


            loss_meter.update(loss.item(), img.shape[0])
            loss_momentum_contrast.update(momentum_contrast.item(), img.shape[0])
            loss_info_nce.update(info_nce.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f} Loss_momentum_contrast {:.3f}, Loss_info_nce {:.3f}; Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, loss_momentum_contrast.avg, loss_info_nce.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

                torch.save(memory.features,
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'memory' + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            
            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.cuda()
                    camids = camids.cuda()
                    target_view = target_view.cuda()
                    feat1, feat2 = model(img, memory_features=memory.features)
                    evaluator1.update((feat1, vid, camid))
                    evaluator2.update((feat2, vid, camid))
            print('Global feature evaluation:')
            cmc, mAP, _, _, _, _, _ = evaluator1.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()

            print('Reconstructed feature evaluation:')
            cmc, mAP, _, _, _, _, _ = evaluator2.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query, 
                 memory):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model.to(device))

    model.eval()
    img_path_list = []

    temp_list = []
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, memory_features=memory)

            
            for i in range(img.size(0)):
                if imgpath[i] == '0035_c1_f0057552.jpg':
                    temp_list.append(feat[i])
                if imgpath[i] == '0035_c1_f0057672.jpg':
                    temp_list.append(feat[i])

            if len(temp_list)==2:
                aaa = temp_list[0]
                bbb = temp_list[1]
                aaa = F.normalize(aaa, dim=1)
                bbb = F.normalize(bbb, dim=1)
                print(torch.cosine_similarity(aaa, bbb, dim=1))

                import pdb
                pdb.set_trace()
                print(2)
                # temp = feat[i].unsqueeze(dim=1)
                
                # if imgpath[i] == '0019_c1_f0051350.jpg':
                    
                #     import pdb
                #     pdb.set_trace()
                    
                    
                # utils.save_image(temp, './save_img/' + imgpath[i])
                

            # feat = model(img, )
            # evaluator.update((feat, pid, camid))
            # img_path_list.extend(imgpath)

    # cmc, mAP, _, _, _, _, _ = evaluator.compute()
    # logger.info("Validation Results ")
    # logger.info("mAP: {:.1%}".format(mAP))
    # for r in [1, 5, 10]:
    #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    # return cmc[0], cmc[4]



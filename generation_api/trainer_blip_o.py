import os
from abc import abstractmethod
import json
import time
import torch
import pandas as pd
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from generation_api.tokenizers_blip import Tokenizer


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _train_epoch_blip(self, epoch):
        raise NotImplementedError

    def train(self):
        ## record all the val
        record_json = {}
        not_improved_count = 0
        if self.args.test_best:
            if self.args.tokenizer == 'our':
                result = self._train_epoch(1)
            else:
                result = self._train_epoch_blip(1)
                for key, value in result.items():
                    print('\t{:15s}: {}'.format(str(key), value))
        else:
            for epoch in range(self.start_epoch, self.epochs + 1):
                if self.args.tokenizer == 'our':
                    result = self._train_epoch(epoch)
                else:
                    result = self._train_epoch_blip(epoch)

                # save logged informations into log dict
                log = {'epoch': epoch}
                log.update(result)
                self._record_best(log)
                record_json[epoch] = log

                self._save_file(record_json)

                # print logged informations to the screen
                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                best = False
                if self.mnt_mode != 'off':
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                            self.early_stop))
                        break
                self._print_best()
                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch, save_best=best)
            self._print_best()
            self._print_best_to_file()
            self._save_file(record_json)

    def _save_file(self, log):
        if not os.path.exists(self.args.record_dir):
            os.mkdir(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir,
                                   self.args.dataset_name + '_' + self.args.save_dir.split('/')[2] + '.json')
        with open(record_path, 'w') as f:
            json.dump(log, f)

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir,
                                   self.args.dataset_name + '_' + self.args.save_dir.split('/')[2] + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader, tokenizer):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, tokenizer)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        ## check the training
        self.writer = SummaryWriter()

    def _train_epoch(self, epoch):

        train_loss = 0
        print_loss = 0
        if not self.args.test_best:
            self.model.train()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):

                images = images.to(self.device)

                # output = self.model(images, reports_ids, mode='train')
                # loss = self.criterion(output, reports_ids, reports_masks)
                loss_ita, loss_itm, loss_lm = self.model(images, reports_ids, reports_masks)
                loss = loss_ita + loss_itm + loss_lm

                train_loss += loss.item()
                self.writer.add_scalar("data/Loss", loss.item(), batch_idx + len(self.train_dataloader) * (epoch - 1))
                # To activate the tensorboard: tensorboard --logdir=runs --bind_all
                print_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                if batch_idx % 5 == 0:
                    print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, print_loss / 5))
                    print_loss = 0
            log = {'train_loss': train_loss / len(self.train_dataloader)}
            print("Finish Epoch {} Training, Start Eval...".format(epoch))
        else:
            log = {}

        # if self.args.test_best:
        #     best_state_dict = torch.load(os.path.join(self.args.save_dir,'model_best.pth'), map_location={'cuda:0':'cuda:3'})
        #     self.model.load_state_dict(best_state_dict, strict=False)
        # a = (0, 0)
        # b = ('generate_report', 'ground_truth')
        # d = dict(zip(b, a))
        # item = []

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            # m=0
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                # output = self.model(images, mode='sample')
                # reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())  # 保留tokenizer
                # print(ground_truths)
                reports = self.model.generate(images, sample=False, num_beams=3,
                                              max_length=90,
                                              min_length=5)
                # cur_val_score = self.metric_ftns({m:[ground_truths]}, {m:[reports]})
                # m=m+1
                # if epoch==9 and self.args.dataset_name == 'iu_xray':
                #     d[b[0]]=reports
                #     d[b[1]]=ground_truths
                #     item.append(d.copy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
        self.writer.add_scalar("data/b1/val", val_met['BLEU_1'], epoch)
        self.writer.add_scalar("data/b2/val", val_met['BLEU_2'], epoch)
        self.writer.add_scalar("data/b3/val", val_met['BLEU_3'], epoch)
        self.writer.add_scalar("data/b4/val", val_met['BLEU_4'], epoch)
        self.writer.add_scalar("data/met/val", val_met['METEOR'], epoch)
        self.writer.add_scalar("data/rou/val", val_met['ROUGE_L'], epoch)
        self.writer.add_scalar("data/cid/val", val_met['CIDER'], epoch)

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                # output = self.model(images, mode='sample')
                # reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                reports = self.model.generate(images, sample=False, num_beams=3,
                                              max_length=90,
                                              min_length=5)
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
        self.writer.add_scalar("data/b1/test", test_met['BLEU_1'], epoch)
        self.writer.add_scalar("data/b2/test", test_met['BLEU_2'], epoch)
        self.writer.add_scalar("data/b3/test", test_met['BLEU_3'], epoch)
        self.writer.add_scalar("data/b4/test", test_met['BLEU_4'], epoch)
        self.writer.add_scalar("data/met/test", test_met['METEOR'], epoch)
        self.writer.add_scalar("data/rou/test", test_met['ROUGE_L'], epoch)
        self.writer.add_scalar("data/cid/test", test_met['CIDER'], epoch)

        self.lr_scheduler.step()
        self.writer.close()

        return log

    def _train_epoch_blip(self, epoch):

        train_loss = 0
        print_loss = 0
        if not self.args.test_best:
            self.model.train()
            for batch_idx, (images, captions, knowledge_skg, knowledge_tc) in enumerate(self.train_dataloader):
                images = images.to(self.device)

                # ramp up alpha in the first 2 epochs
                if epoch > 0:
                    alpha = 0.4
                else:
                    alpha = 0.4 * min(1, batch_idx / len(self.train_dataloader))
                loss_ita, loss_itm, loss_lm = self.model(images, captions, alpha)
                loss = loss_ita + loss_itm + loss_lm

                train_loss += loss.item()
                self.writer.add_scalar("data/Loss", loss.item(), batch_idx + len(self.train_dataloader) * (epoch - 1))
                # To activate the tensorboard: tensorboard --logdir=runs --bind_all
                print_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                if batch_idx % 5 == 0:
                    print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, print_loss / 5))
                    print_loss = 0
            log = {'train_loss': train_loss / len(self.train_dataloader)}
            print("Finish Epoch {} Training, Start Eval...".format(epoch))

        else:
            log = {}
            if self.args.have_know:
                self.model.train()
                for batch_idx, (images, captions, knowledge_skg, knowledge_tc) in enumerate(self.train_dataloader):
                    text = self.model.tokenizer(captions, padding='longest', truncation=True, max_length=40,
                                                return_tensors="pt").to(self.device)
                    text_output = self.model.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                                          return_dict=True, mode='text')
                    text_feat = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
                    self.model.create_knowledge(self.device, text_feat, knowledge_tc)
            else:
                text_queue = None
                knowledge_input_ids_queue = None
                knowledge_attention_mask_queue = None

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            a = (0, 0, 0, 0)
            b = ('image_path', 'ground_truth', 'predict')
            d_val = dict(zip(b, a))
            item_val = []
            for batch_idx, (images, captions, knowledge, image_path) in enumerate(self.val_dataloader):
                images = images.to(self.device)

                # output = self.model(images, mode='sample')
                # reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = captions
                reports, knowledge_used = self.model.generate(images, knowledge, sample=False, num_beams=3,
                                                              max_length=90, min_length=5)
                if self.args.test_best:
                    d_val['image_path'] = image_path
                    d_val['predict'] = reports
                    d_val['ground_truth'] = ground_truths
                    # d_val['knowledge_used'] = knowledge_used
                    item_val.append(d_val.copy())

                val_res.extend(reports)
                val_gts.extend(ground_truths)

            # if self.args.test_best:
            #     with open('./records/generation_result/val_generation_result.json', 'w') as file:
            #         file.write(json.dumps(item_val))

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
        self.writer.add_scalar("data/b1/val", val_met['BLEU_1'], epoch)
        self.writer.add_scalar("data/b2/val", val_met['BLEU_2'], epoch)
        self.writer.add_scalar("data/b3/val", val_met['BLEU_3'], epoch)
        self.writer.add_scalar("data/b4/val", val_met['BLEU_4'], epoch)
        self.writer.add_scalar("data/met/val", val_met['METEOR'], epoch)
        self.writer.add_scalar("data/rou/val", val_met['ROUGE_L'], epoch)
        self.writer.add_scalar("data/cid/val", val_met['CIDER'], epoch)

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            a = (0, 0, 0, 0)
            b = ('image_path', 'ground_truth', 'predict')
            d_test = dict(zip(b, a))
            item_test = []
            for batch_idx, (images, captions, knowledge, image_path) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                reports, knowledge_used = self.model.generate(images, knowledge, sample=False, num_beams=3,
                                                              max_length=90, min_length=5)
                ground_truths = captions

                if self.args.test_best:
                    d_test['image_path'] = image_path
                    d_test['ground_truth'] = ground_truths
                    d_test['predict'] = reports
                    # d_test['knowledge_used'] = knowledge_used
                    item_test.append(d_test.copy())

                test_res.extend(reports)
                test_gts.extend(ground_truths)

            # if self.args.test_best:
            #     with open('./records/generation_result/test_generation_result.json', 'w') as file:
            #         file.write(json.dumps(item_test))

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
        self.writer.add_scalar("data/b1/test", test_met['BLEU_1'], epoch)
        self.writer.add_scalar("data/b2/test", test_met['BLEU_2'], epoch)
        self.writer.add_scalar("data/b3/test", test_met['BLEU_3'], epoch)
        self.writer.add_scalar("data/b4/test", test_met['BLEU_4'], epoch)
        self.writer.add_scalar("data/met/test", test_met['METEOR'], epoch)
        self.writer.add_scalar("data/rou/test", test_met['ROUGE_L'], epoch)
        self.writer.add_scalar("data/cid/test", test_met['CIDER'], epoch)

        self.lr_scheduler.step()
        self.writer.close()

        return log

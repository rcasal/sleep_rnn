######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.
import torch
import time
import copy
from global_parameters import device, dtype_target, dtype_data
from model import ce_loss, statistics

def train_model(model, optimizer, dataloaders, scheduler, num_epochs=25):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print('-' * 118)
    print('|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|'.format(
        'Epoch', 'Phase', 'mini_batch', 'loss', 'acc', 'se', 'sp', 'pre', 'npv'))
    print('-' * 118)

    for epoch in range(num_epochs):
        print('-' * 118)

        # Each epoch has a training and validation phase: by default all the modules are initialized to train
        # mode(self.training = True). Also be aware that some layers have different behavior during train and evaluation
        # (like BatchNorm, Dropout) so setting it matters.

        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0.0
            running_se = 0.0
            running_sp = 0.0
            running_pre = 0.0
            running_npv = 0.0

            # Iterate over data.
            for i_batch, sample in enumerate(dataloaders[phase]):

                # sample to GPU
                sample['feat'] = sample['feat'].type(dtype=dtype_data).to(device)
                sample['target'] = sample['target'].type(dtype=dtype_target).to(device)
                # sample['lengths'] = sample['lengths'].to(device)

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(sample['feat'], sample['lengths'])

                    # Compute and print loss.
                    loss = ce_loss(outputs, sample['target'])
                    batch_acc, batch_se, batch_sp, batch_pre, batch_npv = statistics(outputs, sample['target'],
                                                                                      sample['lengths'])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # compute gradient of the loss with respect to model parameters
                        loss.backward()
                        # Calling the step function on an Optimizer makes an update to its parameters
                        optimizer.step()

                # statistics
                running_loss += loss.item() * sample['feat'].size(0)
                running_acc += batch_acc * sample['feat'].size(0)
                running_se += batch_se * sample['feat'].size(0)
                running_sp += batch_sp * sample['feat'].size(0)
                running_pre += batch_pre * sample['feat'].size(0)
                running_npv += batch_npv * sample['feat'].size(0)


                # prints
                if ( (i_batch + 1) % 250 == 0 and i_batch != 0 and phase == 'train'):
                    cte_batch = (i_batch + 1)*sample['feat'].size(0)
                    print('|{:^12}|{:^12}|{:^12}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|'.format(
                        epoch, phase, i_batch, running_loss/cte_batch, running_acc/cte_batch, running_se/cte_batch,
                        running_sp/cte_batch, running_pre/cte_batch, running_npv/cte_batch))


            cte_epoch = (i_batch + 1) * sample['feat'].size(0)
            epoch_loss = running_loss / cte_epoch                   # Recordar lo del drop_last
            epoch_acc = running_acc / cte_epoch
            epoch_se = running_se / cte_epoch
            epoch_sp = running_sp / cte_epoch
            epoch_pre = running_pre / cte_epoch
            epoch_npv = running_npv / cte_epoch

            print('-' * 118)
            print('|{:^12}|{:^12}|{:^12}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|'.format(
                epoch, phase, i_batch, epoch_loss, epoch_acc, epoch_se, epoch_sp, epoch_pre, epoch_npv))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


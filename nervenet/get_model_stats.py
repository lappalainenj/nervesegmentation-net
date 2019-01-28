import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

def get_model_stats(model, test_loader):

    print('Binary out is %s'%model.binary_out)

    sensitivity, specificity = [0, 0]
    stats = {'true_positives': 0, 'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0}
    test_scores = []
    for i, (inputs, targets) in enumerate(tqdm(test_loader)):
        
        inputs = Variable(inputs)
        
        if model.is_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        target = targets['main'].cpu().numpy()

        outputs = model(inputs)
        pred = outputs['main']
        _, pred = torch.max(pred, 1)

        pred = pred.squeeze().data.cpu().numpy()

        if model.binary_out:
            _, binary = torch.max(outputs['binary'], 1)
            if binary.data.numpy() == 0:
                pred = pred * 0  
        
        target_present = 1 in target
        pred_present = 1 in pred

        if target_present and pred_present: #correctly predicted presence of a nerve
            stats['true_positives'] += 1 
        elif not target_present and pred_present: #wrongly predicted presence of a nerve
            stats['false_positives'] += 1
        elif not target_present and not pred_present: #correctly predicted absence of a nerve
            stats['true_negatives'] += 1
        elif target_present and not pred_present: #wrongly predicted absence of a nerve
            stats['false_negatives'] += 1
        
        #print(np.shape(target), np.shape(pred))
        test_scores.append(dice_coefficient(np.squeeze(target), pred))
        

    if((stats['true_positives'] + stats['false_negatives']) > 0):
        sensitivity = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives'])
    if((stats['true_negatives'] + stats['false_positives']) > 0):
        specificity = stats['true_negatives'] / (stats['true_negatives'] + stats['false_positives'])
    stats['sensitivity'] = sensitivity
    stats['specificity'] = specificity
    stats['test_scores'] = test_scores
    stats['accuracy'] = np.mean(test_scores)
    bestscore, index = highscore(test_scores)
    stats['highscore'] = bestscore
    stats['highscore_index'] = index
    print('Done')
    return stats


def dice_coefficient(ground_truth, predicted):
    gt = ground_truth
    p = predicted
    if np.sum(p) + np.sum(gt) == 0:
        return 1
    else:
        dice = np.sum(p[gt==1])*2.0 / (np.sum(p*p) + np.sum(gt*gt))
        return dice
    
def highscore(test_scores):
    x = np.array(test_scores)
    x[x==1] = 0
    score, index = np.max(x), np.argmax(x)
    return score, index


    
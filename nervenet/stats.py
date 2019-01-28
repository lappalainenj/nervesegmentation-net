model.eval()
model.binary_out = True

sensitivity, specificity = [0, 0]
stats = {'true_positives': 0, 'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0}

for i, (inputs, targets) in enumerate(test_loader):

    if model.is_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    target = targets['main'].cpu().numpy()
    
    outputs = model(inputs)
    pred = outputs['main']
    _, pred = torch.max(pred, 1)
    _, binary = torch.max(outputs['binary'], 1)
    
    pred = pred.squeeze().data.cpu().numpy()
    
    ###!!! Comment in/ Comment out to include/exlude binary classification !!!###
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
    
if((stats['true_positives'] + stats['false_negatives']) > 0):
    sensitivity = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives'])
if((stats['true_negatives'] + stats['false_positives']) > 0):
    specificity = stats['true_negatives'] / (stats['true_negatives'] + stats['false_positives'])

print("Stats %s \n Sensitivity: %s \n Specificity: %s" % (stats, sensitivity, specificity)) 
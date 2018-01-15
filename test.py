import cv2


test_output = pd.read_csv('./data/sample-submission.csv')
for i in range(10000):
    filename = './data/test/'+str(i+1)+'.jpg'; #print(filename)
    X = cv2.imread(filename)
    X = np.expand_dims(X, 0)
    y_pred = model.predict(X)
    y_pred_decoded = decode_y2(y_pred, confidence_thresh=0.4, iou_threshold=0.4, top_k='all', input_coords='centroids', normalize_coords=False, img_height=None, img_width=None)
    
    if len(y_pred_decoded[0])==0: 
    	label = 'unknown'
    	print(label)
	else:
	    box = y_pred_decoded[0][0] 
	    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
	label = label.split(':')[0]
	print(label)=label
	test_output._set_value(i, 'Number', label)
test_output.to_csv("./data/test_output.csv", index=False)

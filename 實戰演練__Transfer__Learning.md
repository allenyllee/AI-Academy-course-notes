# 實戰演練__Transfer__Learning

**實務上常用的遷移學習**
1. Fine-tune
複製模型的參數當作初始參數，再用自己的資料進行微調(借用的參數會被改變)
2. Layer transform
只借用模型部分神經層參數，並在訓練過程中凍結該層參數不更新

使用 Kaggle Dogs vs Cats 資料作範例

樣板程式碼流程
1. 前置作業
2. load data
3. TF-建立靜態圖
4. TF-載入模型參數
5. TF-實際執行模型訓練
6. TF-模型測試

![](https://i.imgur.com/R2ZAZoE.png)


[課程練習題\[kaggle-發芽植物分類\] 參考資料](https://chtseng.wordpress.com/2018/01/19/kaggle-%E7%99%BC%E8%8A%BD%E6%A4%8D%E7%89%A9%E5%88%86%E9%A1%9E/)
[理解Batch Normalization](http://blog.csdn.net/qiusuoxiaozi/article/details/77996309)

**課程解答**

    lr = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(lr)
    
    pred = classifier(conv_output, num_class)
    pred_softmax = tf.nn.softmax(pred)
    
    _, train_loss = sess.run([update, loss], feed_dict={input_img: x[0], y_true: y, 
                                                  is_training: True, lr: model_dict['reduce_lr'].lr})
       
    

**修正講義錯誤**
val_loss = sess.run(loss, feed_dict={input_img: x[0], y_true: y, is_training: False})






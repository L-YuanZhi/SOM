import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os
import data_normalizer_c as dn 

# SOMの結果に基づいて，パイプ画像を自動で正規化しデータセットを作成プログラム．
# 正規化はdata_normalizer.pyの関数を使用している．
# 同時に異なる正規化方法で正規化を行うことができる．
class dataset_pipe:
    
    def __init__(self,load_path,save_path,flip=True,start=None,stop=None,step=None):
        
        self.__load_path = load_path # SOMの結果のフォルダ
        self.__save_path = save_path # 正規化画像を保存するフォルダ
        self.__filp = flip #　Noneでない場合はミラーリングでデータを水増し
        self.__angle_list = [0] # 回転で水増しするための角度リスト．
        if start!=None and stop != None and step != None:
            self.__start = start # 回転を開始する角度
            self.__stop = stop # 回転を止める角度
            self.__step = step # 回転の間隔
            self.__rotate = True # trueである場合は回転で水増し
            self.__Angle_list_gen() # 角度リストを生成
        else:
            self.__rotate=False # 回転しない．
    
    def __Angle_list_gen(self):
        # 角度リストを生成
        self.__angle_list =[]
        for angle in range(self.__start,self.__stop,self.__step):
            self.__angle_list.append(angle)

    def Data_a(self):
    # normalize_circleとmove_averageで正規化を行う．
    # 正規分布正規化と平均値移動の二つのデータセットを作成する．
    
        dir_list = os.listdir(self.__load_path)
        for dirs in dir_list:
            label_list = os.listdir(os.path.join(self.__load_path,dirs))
            for label in label_list:
                if dirs+"_"+label not in os.listdir(os.path.join(self.__save_path,"norm")):
                    if label != "lb":
                        os.mkdir(os.path.join(self.__save_path,"norm",dirs+"_"+label))
                
                if dirs+"_"+label not in os.listdir(os.path.join(self.__save_path,"move")):
                    if label != "lb":
                        os.mkdir(os.path.join(self.__save_path,"move",dirs+"_"+label))


            for label in label_list:
                if label != "lb":
                    for file_name in os.listdir(os.path.join(self.__load_path,dirs,label)):
                        input_image =  plt.imread(os.path.join(self.__load_path,dirs,label,file_name))
                        norm_image = dn.Normalize_circle(input_image,85,20)# 正規分布正規化
                        move_image = dn.move_average(input_image,85)# 平均値移動,mitechと違うのは移動した後に回転する．
                    
                        for angle in self.__angle_list:# 水増し
                            rota_image1 = dn.rotation(norm_image,angle)
                            rota_image2 = dn.rotation(move_image,angle)
                            title = file_name+"_"+str(angle)+".bmp"
                            cv2.imwrite(os.path.join(self.__save_path,"norm",dirs+"_"+label,title),rota_image1)
                            cv2.imwrite(os.path.join(self.__save_path,"move",dirs+"_"+label,title),rota_image2)
                            if self.__filp:
                                title = file_name+"_flip_"+str(angle)+".bmp"
                                cv2.imwrite(os.path.join(self.__save_path,"norm",dirs+"_"+label,title),cv2.flip(rota_image1,1))
                                cv2.imwrite(os.path.join(self.__save_path,"move",dirs+"_"+label,title),cv2.flip(rota_image1,1))

    def Data_mt(self):
        # mitechと同じ方法で正規化，回転した後に平均値移動を行う．
        dir_list = os.listdir(self.__load_path)
        for dirs in dir_list:
            label_list = os.listdir(os.path.join(self.__load_path,dirs))
            if "mt_move" not in os.listdir(self.__save_path):
                os.mkdir(os.path.join(self.__save_path,"mt_move"))

            for label in label_list:
                if dirs+"_"+label not in os.listdir(os.path.join(self.__save_path,"mt_move")):
                    if label != "lb":
                        os.mkdir(os.path.join(self.__save_path,"mt_move",dirs+"_"+label))
            
            for label in label_list:
                if label != "lb":
                    for file_name in os.listdir(os.path.join(self.__load_path,dirs,label)):
                        input_image =  plt.imread(os.path.join(self.__load_path,dirs,label,file_name))
                        
                        for angle in self.__angle_list:
                            rota_image = dn.rotation(input_image,angle)
                            move_image = dn.move_average(rota_image,85)
                            title = file_name+"_"+str(angle)+".bmp"
                            cv2.imwrite(os.path.join(self.__save_path,"mt_move",dirs+"_"+label,title),move_image)
                            
                            if self.__filp:
                                title = file_name+"_flip_"+str(angle)+".bmp"
                                cv2.imwrite(os.path.join(self.__save_path,"mt_move",dirs+"_"+label,title),cv2.flip(move_image,1))

    # def Data_mixc(self):
    #     dir_list = os.listdir(self.__load_path)
    #     for dirs in dir_list:
    #         label_list = os.listdir(os.path.join(self.__load_path,dirs))
    #         if "mixc" not in os.listdir(self.__save_path):
    #             os.mkdir(os.path.join(self.__save_path,"mixc"))

    #         for label in label_list:
    #             if dirs+"_"+label not in os.listdir(os.path.join(self.__save_path,"mixc")):
    #                 if label != "lb":
    #                     os.mkdir(os.path.join(self.__save_path,"mixc",dirs+"_"+label))
            
    #         for label in label_list:
    #             if label != "lb":
    #                 for file_name in os.listdir(os.path.join(self.__load_path,dirs,label)):
    #                     input_image =  plt.imread(os.path.join(self.__load_path,dirs,label,file_name))
                        
    #                     for angle in self.__angle_list:
    #                         rota_image = dn.rotation(input_image,angle)
    #                         move_image = dn.move_average(rota_image,85)
    #                         title = file_name+"_"+str(angle)+".bmp"
    #                         cv2.imwrite(os.path.join(self.__save_path,"mixc",dirs+"_"+label,title),move_image)
                            
    #                         if self.__filp:
    #                             title = file_name+"_flip_"+str(angle)+".bmp"
    #                             cv2.imwrite(os.path.join(self.__save_path,"mixc",dirs+"_"+label,title),cv2.flip(move_image,1))
            


if __name__ == '__main__':

    load_path = "clustering_result/0607_bg/87"
    save_path = "/home/lin/pipe_classifier_by_miTech/05_cnn_v3/dataset_pipe/som_cluster/0607_bg/train_dataset_87"
    dp = dataset_pipe(load_path,save_path,flip=True,start=0,stop=360,step=30)

    dp.Data_a()
    dp.Data_mt() 

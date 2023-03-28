import mtanimal as pydemo
import cv2


class AnimalDetect(object):
    def __init__(self, modelpath="../models"):
        pyanimal_obj = pydemo.MTDLAnimal()
        success = pyanimal_obj.LoadModel(modelpath)
        pyanimal_obj.SetMaxNum(5)

        self.pyanimal_obj = pyanimal_obj

    def get_crop(self, imgpath):
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        self.pyanimal_obj.Run(img)
        num_face = self.pyanimal_obj.GetFaceNum()
        if num_face == 0:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            return img
        
        if num_face == 1:
            rect = self.pyanimal_obj.GetFaceRects(0)
        else:
            max_i = 0
            max_score = 0.0
            for i in range(num_face):
                rect = self.pyanimal_obj.GetFaceRects(i)
                if rect.confidence > max_score:
                    max_i = i
                    max_score = rect.confidence
            rect = self.pyanimal_obj.GetFaceRects(max_i)
            #points = self.pyanimal_obj.GetFacePoints(i)
        img = self.expand_crop(img, rect)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img
    
    def expand_crop(self, img, rect, r=0.7):
        ylarge, xlarge, _ = img.shape
        d = max(rect.h, rect.w)
        centerx = rect.x + 0.5*rect.w
        centery = rect.y + 0.5*rect.h

        xmin = max(0, int(centerx-r*d))
        xmax = min(xlarge, int(centerx+r*d))
        ymin = max(0, int(centery-r*d))
        ymax = min(ylarge, int(centery+r*d))
        
        img = img[ymin:ymax, xmin:xmax, :]
        return img
        
        
        
        
        
        """
            cv2.rectangle(img, (int(rect.x), int(rect.y)), (int(rect.x+rect.w), int(rect.y+rect.h)), (255, 0, 0), 1)
            for j in range(34):
                point = points[j]
                cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 125, 233))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        """
#!/usr/bin/env python

"""
dttasks.py -- problem set tasks for the CS181 decision tree problem set
"""

from os import path

from tfutils import tftask
import dtree

def serialize_tree(dtRoot):
    listSrcDestValue = []
    cNodes = 0
    def node_name(dt,ix):
        if dt.is_node():
            return "Node %d (Split on %d)" % (ix,dt.ixAttr)
        return "Leaf %d (%s)" % (ix,str(dt.fLabel)[0])
    def down(dt,ixParent):
        if dt.is_leaf():
            return
        sParentName = node_name(dt,ixParent)
        for cValue,dtChild in dt.dictChildren.iteritems():
            ixNode = len(listSrcDestValue) + 1
            tplEdge = (sParentName,node_name(dtChild,ixNode),cValue)
            listSrcDestValue.append(tplEdge)
            down(dtChild, ixNode)
    down(dtRoot,0)

    listColor = ["#FF0000", "#00FF00", "#0000FF", "#00FFFF", "#FF00FF",
                 "#FFFF00", "#000000", "#FF8800", "#6600DD", "#000055"]
    listEdge = []
    cMinValue = min([tpl[2] for tpl in listSrcDestValue])
    for src,dest,cValue in listSrcDestValue:
        sColor = listColor[(cValue - cMinValue) % len(listColor)]
        listEdge.append((src,dest,{"color":sColor}))
    return listEdge

def datadir(sPath):
    return path.join(path.dirname(__file__),sPath)

def get_clean_insts():
    return dtree.load_csv_dataset(datadir("data.csv"))

def get_noisy_insts():
    return dtree.load_csv_dataset(datadir("noisy.dat"))

class ExampleLogPlotTask(tftask.ChartTask):
    def task(self):
        listP = []
        listData = []
        for i in map(float,xrange(0,101)):
            listP.append(i/100.0)
            dblEntropy = dtree.compute_entropy(i, 100.0 - i)
            listData.append(dblEntropy)
        return {"chart": {"defaultSeriesType":"line"},
                "title": {"text": "Entropy"},
                "xAxis": {"title":{"text":"p"}},
                "yAxis": {"title": {"text":"entropy"}, "min": 0, "max": 1.1},
                "series": [{"name":"Entropy", "data": zip(listP,listData)}]}
    def get_name(self):
        return "Plot Entropy Curve"
    def get_description(self):
        return "Generate a curve of entropy as a function of probability."
    def get_priority(self):
        return -1

class BcwTreeTask(tftask.GraphTask):
    def task(self):
        listInst = get_clean_insts()
        f = open('view.txt' , 'w+')
        for inst in listInst:
        	f.write(str(inst) + '\n')
        f.close()
        dt = dtree.build_tree(listInst)
        return serialize_tree(dt)
    def get_name(self):
        return "Build BCW Tree"
    def get_description(self):
        return "Build a decision tree for clean (non-noisy) BCW data."
    def get_priority(self):
        return 0

class BcwTrainAccuracy(tftask.ChartTask):
    def task(self):
        listInstClean = get_clean_insts()
        listInstNoisy = get_noisy_insts()
        listData = []
        listNames = ["Clean","Noisy"]
        for listInst,sName in zip([listInstClean,listInstNoisy],
                                  listNames):
            
            dt = dtree.build_tree(listInst)
            tf = dtree.TreeFold(listInst,listInst)
            rslt = dtree.evaluate_classification(tf)
            dblCorrect,dblIncorrect = dtree.weight_correct_incorrect(rslt)
            dblAccuracy = dblCorrect/(dblCorrect + dblIncorrect)
            listData.append(dblAccuracy)
        return {"chart": {"defaultSeriesType":"column"},
                "title": {"text": "Clean vs. Noisy Training Set Accuracy"},
                "xAxis": {"categories": listNames},
                "yAxis": {"title": {"text":"Accuracy"}, "min":0.0, "max":1.0},
                "series": [{"name": "Training Set Accuracy",
                            "data": listData}]}
    def get_name(self):
        return "Measure Cross-Validated ID3 Training Set Accuracy"
    def get_description(self):
        return ("Build an unpruned decision tree for both the clean and noisy "
                "BCW data sets and measure the tree's training set accuracy. "
                "No cross-validation is performed.")
    def get_priority(self):
        return 0.5      

class BcwCrossValidateTask(tftask.ChartTask):
    def get_name(self):
        return "Measure Cross-Validated Performance"
    def get_description(self):
        return ("Build decision trees for clean and noisy BCW data and "
                "evaluate their performance through 10-fold cross validation.")
    def get_priority(self):
        return 1
    def build_depth_yield(self,iDepth):
        def yield_cv_folds(listInst,cFold):
            for cvf in dtree.yield_cv_folds(listInst,cFold):
                cvf.cMaxLevel = iDepth
                yield cvf
        return yield_cv_folds
    def task(self):
        listInstClean = dtree.load_csv_dataset(datadir("data.csv"))
        listInstNoisy = dtree.load_csv_dataset(datadir("noisy.dat"))
        cFold = 10
        listSeries = []
        for sLbl,fxn in [("Unpruned", dtree.yield_cv_folds),
                         ("Pruned", dtree.yield_cv_folds_with_validation),
                         ("Boosted", dtree.yield_boosted_folds),
                         ("Stumps", self.build_depth_yield(1)),
                         ("Depth-2", self.build_depth_yield(2))]:
            try:
                fxnScore = lambda listInst: dtree.cv_score(fxn(listInst,cFold)) 
                listData = [fxnScore(listInstClean),fxnScore(listInstNoisy)]
                dictSeries = {"name": sLbl, "data": listData}
            except NotImplementedError:
                # we can forget about un-implemented functionality
                dictSeries = {"name": sLbl + " (not implemented)", "data":[]}
            listSeries.append(dictSeries)
                
        return {"chart": {"defaultSeriesType":"column"},
                "title": {"text": "Clean vs. Noisy Classification"},
                "xAxis": {"categories": ["Clean", "Noisy"]},
                "yAxis": {"title": {"text": "Fraction Correct"},
                          "min":0.0, "max":1.0},
                "series": listSeries}

class BoostingCoefficients(tftask.ChartTask):
    def get_name(self):
        return "Plot Boosting Classifier Weights"
    def get_description(self):
        return ("Run boosting using decision stumps on clean BCW data, then "
                "plot the weights of the resulting classifiers.")
    def get_priority(self):
        return 4
    def task(self):
        listInst = dtree.load_csv_dataset(datadir("data.csv"))
        br = dtree.boost(listInst)
        return {"chart": {"defaultSeriesType":"line"},
                "title": {"text": "Boosting Classifier Weights"},
                "xAxis": {"title": {"text": "Classifier Number"}},
                "series": [{"name": "Classifier Weights",
                            "data": br.listDblCferWeight}]}

class BcwPrunedDecisionTree(tftask.GraphTask):
    def get_name(self):
        return "Prune BCW Decision Tree"
    def get_description(self):
        return ("Build a decision tree for clean BCW data, "
                "then prune it using a validation set.")
    def get_priority(self):
        return 2
    def task(self):
        listInst = dtree.load_csv_dataset(datadir("data.csv"))
        dt = dtree.build_tree(listInst[:-10])
        dtree.prune_tree(dt,listInst[-10:])
        return serialize_tree(dt)

class BcwDecisionStump(tftask.GraphTask):
    def get_name(self):
        return "Build Decision Stump"
    def get_description(self):
        return ("Build a decision stump (depth 1 decision tree) for clean "
                "BCW data.")
    def get_priority(self):
        return 3
    def task(self):
        listInst = dtree.load_csv_dataset(datadir("data.csv"))
        dt = dtree.build_tree(listInst, cMaxLevel=1)
        return serialize_tree(dt)

class BcwCompareBoostingParameters(tftask.ChartTask):
    def get_name(self):
        return "Compare Boosting Parameters"
    def get_description(self):
        return ("Evaluate the performance of boosting for various numbers "
                "of rounds, and with different weak learners.")
    def get_priority(self):
        return 3.5
    def build_fold_generator(self, cMaxLevel, cMaxRounds):
        def yield_folds(listInst,cFold):
            for cvf in dtree.yield_boosted_folds(listInst,cFold):
                cvf.cMaxLevel = cMaxLevel
                cvf.cMaxRounds = cMaxRounds
                yield cvf
        return yield_folds
    def task(self):
        listInstClean = get_clean_insts()
        listInstNoisy = get_noisy_insts()
        listSeries = []
        cFold = 10
        for sName,cMaxLevel,cMaxRounds in [("Depth 1, 10 Rounds", 1, 10),
                                           ("Depth 2, 10 Rounds", 2, 10),
                                           ("Depth 1, 30 Rounds", 1, 30),
                                           ("Depth 2, 30 Rounds", 2, 30)]:
            fxnGen = self.build_fold_generator(cMaxLevel,cMaxRounds)
            fxnScore = lambda listInst: dtree.cv_score(fxnGen(listInst,cFold))
            listData = [fxnScore(listInstClean),fxnScore(listInstNoisy)]
            listSeries.append({"name":sName, "data": listData})
            
        sTitle = "Classification Accuracy For Different Boosting Parameters"
        return {"chart": {"defaultSeriesType":"column"},
                "title": {"text": sTitle},
                "xAxis": {"categories": ["Clean", "Noisy"]},
                "yAxis": {"title": {"text": "Fraction Correct"},
                          "min":0.0, "max":1.0},
                "series": listSeries}

class BcwBoostingTrainVsTest(tftask.ChartTask):
    def get_name(self):
        return "Compare Boosting Training- and Test-Set Accuracy"
    def get_description(self):
        return ("Assess the relationship in boosting between cross-validated "
                "training- and test-set performance on clean BCW data.")
    def build_fold_gen(self, cRounds, fUseTraining):
        def yield_folds(listInst,cFold):
            for cvf in dtree.yield_boosted_folds(listInst,cFold):
                cvf.cMaxRounds = cRounds
                if fUseTraining:
                    cvf.listInstTest = cvf.listInstTraining
                yield cvf
        return yield_folds
    def get_priority(self):
        return 5
    def task(self):
        listInst = get_clean_insts()
        cFold = 10
        listSeries = []
        for sNamePref,fUseTraining in [("Training", True), ("Test", False)]:
            listData = []
            for cRounds in xrange(1,16):
                fxnGen = self.build_fold_gen(cRounds,fUseTraining)
                listData.append(dtree.cv_score(fxnGen(listInst,cFold)))
            listSeries.append({"name": sNamePref + " Set Accuracy",
                               "data": listData})
        return {"chart": {"defaultSeriesType": "line"},
                "title": {"text": "Training- vs. Test-Set Accuracy"},
                "xAxis": {"min": 0, "max":16, "title": {"text":"Rounds"}},
                "yAxis": {"title": {"text":"Accuracy"}},
                "series": listSeries}
                

                

if __name__ == "__main__":
    btt = BcwTreeTask()
    print btt.task()
    print tftask.list_tasks(BcwTreeTask.__module__)
        

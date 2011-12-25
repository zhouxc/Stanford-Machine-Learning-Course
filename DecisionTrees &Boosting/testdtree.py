#!/usr/bin/env python

import functools
import random
import math
import unittest

import dtree  

def repeated(fn):
    @functools.wraps(fn)
    def wrapper(obj, *args, **kwargs):
        cRepeat = getattr(obj,"REPEAT") if hasattr(obj,"REPEAT") else 100
        for _ in xrange(cRepeat):
            fn(obj,*args,**kwargs)
    wrapper.wrapped = fn
    return wrapper

def randbool(dblP=0.5):
    return random.random() < dblP

def randlist(lo,hi,n):
    return map(lambda x: x(lo,hi), [random.randint]*n)

def build_one_instance(cAttrs,cValues,fxnGenWeight,fxnGenLabel):
    listAttrs = randlist(0,cValues-1,cAttrs)
    return dtree.Instance(listAttrs, fxnGenLabel(listAttrs), fxnGenWeight())

def build_instance_generator(dblLabelDist=0.5,cAttrs=10, cValues=4,
                             fxnGenWeight=None, fxnGenLabel=None):
    if fxnGenWeight is None:
        fxnGenWeight = lambda: 1.0
    if fxnGenLabel is None:
        fxnGenLabel = lambda _: randbool(dblLabelDist)
    def build_instances(n=1):
        build1 = lambda: build_one_instance(cAttrs,cValues,fxnGenWeight,
                                            fxnGenLabel)
        return [build1() for _ in xrange(n)]
    build_instances.cAttrs = cAttrs
    build_instances.cValues = cValues
    return build_instances

def build_entropy_one_instances(cAttr,cValue):
    listInstTrue = [dtree.Instance([0 for _ in xrange(cAttr)],True)
                    for f in xrange(cValue)]
    listInstFalse = [dtree.Instance([0 for _ in xrange(cAttr)],False,0.5)
                     for f in xrange(2*cValue)]
    for ixAttr in xrange(cAttr):
        for ixValue in xrange(cValue):
            ixFalse = 2*ixValue
            listInstTemp = (listInstTrue[ixValue],
                            listInstFalse[ixFalse],
                            listInstFalse[ixFalse+1])
            for inst in listInstTemp:
                inst.listAttrs[ixAttr] = ixValue
    return listInstTrue + listInstFalse
                
def force_instance_consistency(listInst):
    dictMapping = {}
    for inst in listInst:
        tupleKey = tuple(inst.listAttrs)
        if tupleKey in dictMapping:
            inst.fLabel = dictMapping[tupleKey]
        else:
            dictMapping[tupleKey] = inst.fLabel

def build_consistent_generator(*args, **kwargs):
    fxnGen = build_instance_generator(*args,**kwargs)
    @functools.wraps(fxnGen)
    def wrapper(cInst):
        listInst = fxnGen(cInst)
        force_instance_consistency(listInst)
        return listInst
    return wrapper

def build_jagged_instances():
    return [dtree.Instance([0]*random.randint(5,10))
            for _ in xrange(random.randint(25,30))]

class EntropyTest(unittest.TestCase):
    REPEAT = 100
    cInsts = 100

    @repeated
    def test_compute_entropy(self):
        dblK = 1000000.0*random.random()
        self.assertAlmostEqual(1.0, dtree.compute_entropy(dblK,dblK))
        self.assertAlmostEqual(0.0, dtree.compute_entropy(0.0, dblK))
        self.assertAlmostEqual(0.0, dtree.compute_entropy(dblK, 0.0))
        
    @repeated
    def test_separate_by_attribute(self):
        fxnGen = build_instance_generator(0.5)
        listInst = fxnGen(self.cInsts)
        for ixAttr in xrange(fxnGen.cAttrs):
            dictInst = dtree.separate_by_attribute(listInst, ixAttr)
            setValues = set([inst.listAttrs[ixAttr] for inst in listInst])
            self.assertEqual(len(setValues), len(dictInst))
            for cValue,listInstSeparate in dictInst.iteritems():
                for inst in listInstSeparate:
                    self.assertEqual(cValue, inst.listAttrs[ixAttr])
                    
    @repeated
    def test_compute_entropy_of_split(self):
        cAttrs = random.randint(2,20)
        cValues = random.randint(1,30)
        fxnGenOne = lambda _: build_entropy_one_instances(cAttrs, cValues)
        fxnGenOne.cAttrs = cAttrs
        fxnGenOne.cValues = cValues
        fxnGenZero = build_instance_generator(0.0, cAttrs=3)
        dblDelta = 0.01
        for fxnGen,dblP in zip((fxnGenOne,fxnGenZero,),(1.0,0.0)):
            listInst = fxnGen(self.cInsts)
            for ixAttr in xrange(fxnGen.cAttrs):
                dictInst = dtree.separate_by_attribute(listInst, ixAttr)
                dblEntropy = dtree.compute_entropy_of_split(dictInst)
                self.assertTrue(abs(dblEntropy - dblP) < dblDelta,
                                "%.3f not within %.3f of expected %.3f" %
                                (dblEntropy, dblDelta, dblP))

    def test_compute_entropy_of_split_weighted(self):       
        fxnGenTrue = build_instance_generator(1.0)
        fxnGenFalse = build_instance_generator(0.0, fxnGenWeight=lambda: 0.25)
        cInst = 10
        listInst = fxnGenTrue(cInst) + fxnGenFalse(4*cInst)
        dblEntropy = dtree.compute_entropy_of_split({0: listInst})
        self.assertAlmostEqual(1.0, dblEntropy)

    @repeated
    def test_choose_split_attribute(self):
        cAttrs = 4
        ixBest = random.randint(0,cAttrs-1)
        def generate_label(listAttrs):
            return bool(listAttrs[ixBest] % 2)
        fxnGen = build_instance_generator(cAttrs=cAttrs,
                                          fxnGenLabel=generate_label)
        listInst = fxnGen(self.cInsts)
        ixChosen,dictBest = dtree.choose_split_attribute(range(cAttrs),
                                                         listInst, 0.0)
        self.assertEqual(ixBest,ixChosen)
        # should come up w/something stronger
        self.assertEqual(type(dictBest),dict)

    @repeated
    def test_check_for_common_label(self):
        fxnGenTrue = build_instance_generator(1.0)
        fxnGenFalse = build_instance_generator(0.0)
        fxnGenNone = build_instance_generator()
        listPair = ((fxnGenTrue,True),(fxnGenFalse,False),(fxnGenNone,None),)
        for fxnGen,expected in listPair:
            listInst = fxnGen(self.cInsts)
            fLabel = dtree.check_for_common_label(listInst)
            self.assertTrue(fLabel is expected, "%s is not %s"
                            % (fLabel,expected))

    @repeated
    def test_majority_label(self):
        fxnGenTrue = build_instance_generator(1.0)
        fxnGenFalse = build_instance_generator(0.0)
        cLenTrue = random.randint(5,10)
        cLenFalse = random.randint(5,10)
        if cLenTrue == cLenFalse:
            cLenTrue += 1
        listInst = fxnGenTrue(cLenTrue) + fxnGenFalse(cLenFalse)
        fMajorityLabel = dtree.majority_label(listInst)
        self.assertEqual(fMajorityLabel, cLenTrue > cLenFalse)

    @repeated
    def test_majority_label_weighted(self):
        dblScale = 25.0
        def gen_insts_for_label(fLabel):
            dblW = random.random() * dblScale
            listInst = []
            dblInstWeight = 0.0
            while dblInstWeight < dblW:
                dblNextWeight = random.random()
                listInst.append(dtree.Instance([],fLabel,dblNextWeight))
                dblInstWeight += dblNextWeight
            return listInst,dblInstWeight
        listInstT,dblT = gen_insts_for_label(True)
        listInstF,dblF = gen_insts_for_label(False)
        listInstAll = listInstT + listInstF
        random.shuffle(listInstAll)
        fMajorityLabel = dtree.majority_label(listInstAll)
        self.assertEqual(dblT > dblF, fMajorityLabel)        

def check_dt_members(dt):
    if dt.is_leaf() and dt.is_node():
        return False, ("Tree is not clearly a leaf or node. Only one"
                       " of fLabel and ixAttr should be not None.")
    for cValue,dtChild in dt.dictChildren.iteritems():
        fSuccess,sMsg = check_dt_members(dtChild)
        if not fSuccess:
            return fSuccess,sMsg
    return True,None

class ConstructionTest(unittest.TestCase):
    def check_dt(self,dtRoot,cMaxLevel):
        def down(dt,cLvl):
            self.assertTrue(cLvl <= cMaxLevel)
            if dt.is_node():
                for dtChild in dt.dictChildren.values():
                    down(dtChild,cLvl+1)
        down(dtRoot,0)

    def assert_dt_members(self,dt):
        fSuccess,sMsg = check_dt_members(dt)
        self.assertTrue(fSuccess, sMsg)

    @repeated
    def test_build_tree_rec_leaf(self):
        fLabel = randbool()
        listInst = [dtree.Instance([],fLabel)]*random.randint(1,3)
        dt = dtree.build_tree_rec([],listInst,0.0,-1)
        self.assert_dt_members(dt)
        self.assertTrue(dt.is_leaf(), "dt was not a leaf")
        self.assertEqual(dt.fLabel, fLabel)

    @repeated
    def test_build_tree_rec_stump(self):
        pairBounds = (5,10)
        build_list_inst_bool = (lambda f:
            [dtree.Instance([int(f),randbool()],fLabel=f)
             for _ in xrange(random.randint(*pairBounds))])
        listInst = build_list_inst_bool(True) + build_list_inst_bool(False)
        setIxAttr = set(range(2))
        cPrevSetIxAttrLen = len(setIxAttr)
        dt = dtree.build_tree_rec(setIxAttr, listInst, 0.0,-1)
        self.assert_dt_members(dt)
        self.assertEqual(cPrevSetIxAttrLen, len(setIxAttr),
                         "setIxAttr changed size in build_tree_rec")
        self.assertTrue(dt.is_node(), "dt was not a node")
        self.assertEqual(dt.ixAttr, 0)
        dt0 = dt.dictChildren[0]
        dt1 = dt.dictChildren[1]
        for dtChild,fExpected in ((dt0,False), (dt1,True)):
            self.assertTrue(dtChild.is_leaf(), "dtChild was not a leaf")
            self.assertEqual(dtChild.fLabel, fExpected)

    @repeated
    def test_build_tree_depth_limit(self):
        fxnGen = build_consistent_generator(10)
        listInst = fxnGen(100)
        cMaxLevel = random.randint(0,3)
        dt = dtree.build_tree(listInst, cMaxLevel=cMaxLevel)
        self.assert_dt_members(dt)
        self.check_dt(dt,cMaxLevel)

    @repeated
    def test_build_tree_gain_limit(self):
        listInst = []
        cAttr = random.randint(5,10)
        ixAttrImportant = random.randint(0,cAttr-1)
        for _ in xrange(random.randint(25,150)):
            listAttr = randlist(0,1,cAttr)
            fLabel = bool(listAttr[ixAttrImportant])
            listInst.append(dtree.Instance(listAttr,fLabel))
        dt = dtree.build_tree(listInst, dblMinGain=0.55)
        self.assert_dt_members(dt)
        self.assertTrue(dt.is_node())
        self.check_dt(dt,1)        

    @repeated
    def test_count_instance_attributes(self):
        cLen = random.randint(3,10)
        listInst = [dtree.Instance([0]*cLen)]*random.randint(5,10)
        cLenObserved = dtree.count_instance_attributes(listInst)
        self.assertEqual(cLen, cLenObserved)
        listInstJag = build_jagged_instances()
        self.assertTrue(dtree.count_instance_attributes(listInstJag) is None)

    def test_build_tree_raises(self):
        self.assertRaises(TypeError, dtree.build_tree,
                          build_jagged_instances())
    @repeated
    def test_build_tree(self):
        # test case size grows exponentially in this
        cAttrs = random.randint(1,5)
        listInst = []
        for ixAttr in xrange(cAttrs):
            cEach = 2**(cAttrs - ixAttr)
            listAttrPrefixLeft = [1]*ixAttr
            for _ in xrange(cEach):
                listAttrSuffix = [0]*(cAttrs - ixAttr)
                listAttr = listAttrPrefixLeft + listAttrSuffix
                fLabel = bool(ixAttr % 2)
                inst = dtree.Instance(listAttr,fLabel)
                listInst.append(inst)
        dt = dtree.build_tree(listInst)
        for ixAttr in xrange(cAttrs-1):
            self.assertEqual(dt.ixAttr, ixAttr)
            dtLeft = dt.dictChildren[0]
            self.assertTrue(dtLeft.is_leaf())
            self.assertEqual(dtLeft.fLabel, bool(ixAttr % 2))
            dt = dt.dictChildren[1]
        self.assertTrue(dt.is_leaf())
        self.assertEqual(dt.fLabel, not (cAttrs % 2))

    @repeated
    def test_build_tree_no_gain(self):
        listAttr = randlist(0,5,10)
        listInst = [dtree.Instance(listAttr, randbool())]*random.randint(25,30)
        dt = dtree.build_tree(listInst)
        fMajorityLabel = dtree.majority_label(listInst)
        self.assertTrue(dt.is_leaf())
        self.assertEquals(dt.fLabel, fMajorityLabel)        

def build_random_tree(cAttr,cValue):
    def down(listIxAttr):
        if listIxAttr:
            ixAttr = random.choice(listIxAttr)
            listIxAttrNext = list(listIxAttr)
            listIxAttrNext.remove(ixAttr)
            dt = dtree.DTree(ixAttr=ixAttr,fDefaultLabel=randbool())
            for cV in xrange(cValue):
                dt.add(down(listIxAttrNext), cV)
            return dt
        return dtree.DTree(fLabel=randbool())
    return down(range(cAttr))

def build_random_instance_from_dt(dt,cAttr=None):
    listPath = []
    while dt.is_node():
        cV,dtChild = random.choice(dt.dictChildren.items())
        listPath.append((dt.ixAttr,cV))
        dt = dtChild
    assert dt.is_leaf()
    listAttr = []
    cMaxAttr = max([ixAttr for ixAttr,_ in listPath])
    dictPath = dict(listPath)
    if cAttr is None:
        cAttr = cMaxAttr + random.randint(1,5)
    for ixAttr in xrange(cAttr):
        cV = dictPath[ixAttr] if ixAttr in dictPath else random.randint(0,10)
        listAttr.append(cV)
    return dtree.Instance(listAttr, dt.fLabel),listPath
        
class PredictionTest(unittest.TestCase):
    @repeated
    def test_classify(self):
        dt = build_random_tree(4,3)
        for _ in xrange(5):
            inst,listPath = build_random_instance_from_dt(dt)
            fLabel = dtree.classify(dt,inst)
            self.assertEqual(inst.fLabel, fLabel)

    @repeated
    def test_classify_unknown(self):
        cValue = 3
        dt = build_random_tree(4,cValue)
        inst = dtree.Instance(randlist(cValue+1, cValue+5, 4))
        fLabel = dtree.classify(dt,inst)
        self.assertEqual(fLabel, dt.fDefaultLabel)        

def check_instance_membership(listInstDb, listInstQueries):
    def make_key(inst):
        return tuple(inst.listAttrs + [inst.fLabel])
    setDb = set(map(make_key, listInstDb))
    for inst in listInstQueries:
        tupleKey = make_key(inst)
        if tupleKey not in setDb:
            return False
    return True

class EvaluationTest(unittest.TestCase):
    REPEAT = 25
    
    @repeated
    def test_evaluate_classification(self):
        def increase_values(inst):
            listIncreased = [c+cValues+1 for c in inst.listAttrs]
            return dtree.Instance(listIncreased, not fMajorityLabel)
        def filter_unclassifiable(listInst):
            dt = dtree.build_tree(listInst)
            return [inst for inst in listInst
                    if dtree.classify(dt,inst) == inst.fLabel]
        cValues = 2
        fxnGen = build_instance_generator(cValues=cValues)
        listInst = fxnGen(15)
        force_instance_consistency(listInst)
        listInst = filter_unclassifiable(listInst)
        fMajorityLabel = dtree.majority_label(listInst)
        listInstImpossible = map(increase_values,listInst)
        listInstTest = listInst + listInstImpossible
        cvf = dtree.TreeFold(listInst, listInstTest)
        rslt = dtree.evaluate_classification(cvf)
        self.assertEqual(len(listInst), len(rslt.listInstCorrect))
        self.assertEqual(len(listInstImpossible), len(rslt.listInstIncorrect))
        self.assertTrue(check_instance_membership(
            listInst, rslt.listInstCorrect), "Missing correct instances")
        self.assertTrue(check_instance_membership(
            listInstImpossible, rslt.listInstIncorrect),
                        "Missing incorrect instances")

    @repeated
    def test_weight_corrrect_incorrect(self):
        def make_list(cLen):
            listI = []
            dblSum = 0.0
            for _ in xrange(cLen):
                dbl = math.exp(-random.random() - 0.1) * 10.0
                listI.append(dtree.Instance([],randbool(),dbl))
                dblSum += dbl
            return listI,dblSum
        listInstCorrect,dblCorrect = make_list(random.randint(0,10))
        listInstIncorrect,dblIncorrect = make_list(random.randint(0,10))
        rslt = dtree.EvaluationResult(listInstCorrect, listInstIncorrect,None)
        dblC,dblI = dtree.weight_correct_incorrect(rslt)
        self.assertAlmostEqual(dblCorrect,dblC)
        self.assertAlmostEqual(dblIncorrect,dblI)

def build_foldable_instances(lo=3,hi=10):
    cFold = random.randint(lo,hi)
    cInsts = random.randint(1,10)*cFold
    return [dtree.Instance([i],randbool()) for i in range(cInsts)],cFold

def build_folded_set(listInst):
    return set([inst.listAttrs[0] for inst in listInst])

def is_valid_cvf_builder(obj, fxnBuildCvf, fxnCheckEach, fUseValidation):
    listInst,cFold = build_foldable_instances()
    cFoldSize = len(listInst)/cFold
    setI = build_folded_set(listInst)
    cFoldsYielded = 0
    for cvf in fxnBuildCvf(list(listInst),cFold):
        if not fxnCheckEach(cvf):
            return False
        setTrain = build_folded_set(cvf.listInstTraining)
        setTest = build_folded_set(cvf.listInstTest)
        setValidation = (build_folded_set(cvf.listInstValidate)
                         if fUseValidation else set())
        obj.assertEqual(cFoldSize, len(setTest))
        if fUseValidation:
            obj.assertEqual(cFoldSize, len(setTest))
            cFoldsInTraining = cFold - 2
        else:
            cFoldsInTraining = cFold - 1
        obj.assertEqual(cFoldSize*cFoldsInTraining, len(setTrain))
        obj.assertEqual(setI - setTrain - setValidation, setTest)
        obj.assertEqual(setI - setTest - setValidation, setTrain)
        obj.assertEqual(setI - setTrain - setTest, setValidation)
        cFoldsYielded += 1
    return cFold == cFoldsYielded

class CrossValidationTest(unittest.TestCase):
    REPEAT = 15
    
    @repeated
    def test_yield_cv_folds(self):
        fxnCheck = lambda cvf: isinstance(cvf, dtree.TreeFold)
        is_valid_cvf_builder(self, dtree.yield_cv_folds, fxnCheck,False)
        
    @repeated
    def test_cv_score(self):
        def label_weight(listInst, fLabel):
            dblWeight = 0.0
            for inst in listInst:
                if inst.fLabel == fLabel:
                    dblWeight += inst.dblWeight
            return dblWeight
        cValues = 4
        fxnGen = build_consistent_generator(cValues=cValues,
                                            fxnGenWeight=random.random)
        cInst = random.randint(30,60)
        listLeft = fxnGen(cInst)
        listRight = [dtree.Instance([cAttr+cValues+1
                                     for cAttr in inst.listAttrs],
                              inst.fLabel) for inst in fxnGen(cInst)]
        fMajL = dtree.majority_label(listLeft)
        fMajR = dtree.majority_label(listRight)
        iterableFolds = [dtree.TreeFold(listLeft,listRight),
                         dtree.TreeFold(listRight,listLeft)]
        dblScore = dtree.cv_score(iterableFolds)
        dblL = label_weight(listRight, fMajL)
        dblR = label_weight(listLeft, fMajR)
        dblTotalWeight = sum([inst.dblWeight for inst in listRight + listLeft])
        self.assertAlmostEqual((dblL + dblR)/dblTotalWeight, dblScore)

    @repeated
    def test_yield_cv_folds_with_validation(self):
        fxnCheck = lambda cvf: isinstance(cvf, dtree.PrunedFold)
        is_valid_cvf_builder(self, dtree.yield_cv_folds_with_validation,
                             fxnCheck, True)

class PruneTest(unittest.TestCase):
    REPEAT = 10
    
    @repeated
    def test_prune_tree(self):
        """
        Test bottom-up pruning with a validation set.

        The test builds a random tree, then randomly chooses a node at which
        to prune. To induce pruning, the test does the following:
        - set the default label of the node to T
        - set the default label of the nodes, and actual label of the leaves,
          of all descendants to F
        - generate a large number of T instances that follow a path
          through the node
        - set the default labels of all ancestors of the node to F
        - prune the tree
        - repeat for the node's parent, continuing up to the root.
        """
        def set_labels(dtRoot,f):
            def down(dt):
                if dt.is_leaf():
                    dt.fLabel = f
                dt.fDefaultLabel = f
                map(down,dt.dictChildren.values())
            down(dtRoot)
        def check_passes(dtRoot,dtCheck,inst):
            def down(dt):
                assert not dt.is_leaf()
                assert len(dt.dictChildren) == cValue
                dt = dt.dictChildren[inst.listAttrs[dt.ixAttr]]
                if dt == dtCheck:
                    return
            down(dtRoot)

        cAttr = random.randint(2,4)
        cValue = random.randint(2,4)
        dtBase = build_random_tree(cAttr,cValue)
        listPath = []
        listAttrs = []
        listDt = []
        fTargetValue = True#randbool()
        set_labels(dtBase, not fTargetValue)
        dt = dtBase    
        while not dt.is_leaf():
            ixValue = random.choice(dt.dictChildren.keys())
            listPath.append(ixValue)
            listAttrs.append(dt.ixAttr)
            #print ixValue
            dt = dt.dictChildren[ixValue]
        #print "-----------------------"

        while listPath:
            listPath.pop()
            dt = dtRoot = dtBase
            for ixValue in listPath:
                #print ixValue
                dt = dt.dictChildren[ixValue]
                assert dt.is_node()
            #print "-----------------------------------"
            dt.fDefaultLabel = fTargetValue
            listInst = []
            fxnEnd = lambda: randlist(0,cValue-1,cAttr - len(listPath))
            for _ in xrange(random.randint(1,10)):
                listValue = listPath + fxnEnd()
                listInstAttr = [None for _ in xrange(cAttr)]
                assert len(listValue) == cAttr
                for ixValue,ixAttr in zip(listValue,listAttrs):
                    listInstAttr[ixAttr] = ixValue
                inst = dtree.Instance(listInstAttr, fTargetValue)
                check_passes(dtRoot,dt,inst)
                listInst.append(inst)
            dtree.prune_tree(dtRoot,listInst)
            dt = dtRoot
            for ix,ixValue in enumerate(listPath):
                assert dt.ixAttr == listAttrs[ix]
                self.assertTrue(dt.is_node(), str(dtRoot))
                self.assertTrue(ixValue in dt.dictChildren)
                dt = dt.dictChildren[ixValue]
            self.assertTrue(dt.is_leaf(), str(dt))
        
def is_stump(dt):
    for cV,dtChild in dt.dictChildren.iteritems():
        if not dtChild.is_leaf():
            return False
    return True

fxnRandomWeight = lambda: random.random()*1000.0 + 0.1
build_random_weight = build_instance_generator(fxnGenWeight=fxnRandomWeight)

class BoostTest(unittest.TestCase):
    REPEAT = 10
        
    @repeated
    def test_normalize_weights(self):
        cInst = 100
        listInst = build_random_weight(cInst)
        def weight_sum():
            return sum([inst.dblWeight for inst in listInst], 0.0)
        self.assertTrue(weight_sum() > 1.0)
        dtree.normalize_weights(listInst)
        self.assertAlmostEqual(1.0, weight_sum())

    @repeated
    def test_init_weights(self):
        cInst = 100
        listInst = build_random_weight(cInst)
        dtree.init_weights(listInst)
        for inst in listInst:
            self.assertAlmostEqual(1.0/float(cInst), inst.dblWeight)

    @repeated
    def test_classifier_error(self):
        cInst = 100
        listInst = build_instance_generator()(cInst)
        ix = random.randint(0,cInst)
        rslt = dtree.EvaluationResult(listInst[:ix], listInst[ix:], None)
        self.assertAlmostEqual(float(cInst-ix)/float(cInst),
                               dtree.classifier_error(rslt))
        
    @repeated
    def test_classifier_weight(self):
        dblError = random.random()
        dblWeight = dtree.classifier_weight(dblError)
        dblFrac = math.exp(2.0*dblWeight)
        self.assertAlmostEqual(dblError, 1.0/(dblFrac + 1.0))

    @repeated
    def test_update_weight_unnormalized(self):
        dblWeight = random.normalvariate(0.0,1.0)
        dblClassifierWeight = random.normalvariate(0.0,10.0)
        fLabel = randbool()
        fClassifiedLabel = randbool()
        inst = dtree.Instance([],fLabel=fLabel,dblWeight=dblWeight)
        dtree.update_weight_unnormalized(inst, dblClassifierWeight,
                                         fClassifiedLabel)
        dblWeightNew = inst.dblWeight
        dblWeightNew /= dblWeight
        dblWeightNew = math.log(dblWeightNew)
        dblWeightNew /= dblClassifierWeight
        if fLabel == fClassifiedLabel:
            self.assertAlmostEqual(-1.0, dblWeightNew)
        else:
            self.assertAlmostEqual(1.0, dblWeightNew)

    @repeated
    def test_one_round_boost(self):
        fxnGen = build_consistent_generator()
        cInst = 100
        listInst = fxnGen(cInst)
        for inst in listInst:
            inst.listAttrs[0] = int(inst.fLabel)
        listInstIncorrect = random.sample(listInst,cInst/10)
        for inst in listInstIncorrect:
            inst.fLabel = not inst.listAttrs[0]
            inst.dblWeight = 0.1
        dt,dblError,dblCferWeight = dtree.one_round_boost(listInst,1)
        self.assertTrue(is_stump(dt))
        self.assertAlmostEqual(1.0/91.0, dblError)
        self.assertAlmostEqual(dtree.classifier_weight(dblError),
                               dblCferWeight)
        self.assertAlmostEqual(1.0, sum([inst.dblWeight for inst in listInst]))

    @repeated
    def test_boost(self):
        listAttr = randlist(0,5,10)
        listInst = [dtree.Instance(listAttr, True) for _ in xrange(100)]
        listInstFalse = random.sample(listInst,10)
        for inst in listInstFalse:
            inst.fLabel = False
        listInstCopy = [inst.copy() for inst in listInst]
        br = dtree.boost(listInstCopy)
        dblWeightExpected = dtree.classifier_weight(0.1)
        self.assertAlmostEqual(br.listDblCferWeight[0], dblWeightExpected)

    @repeated
    def test_boost_maxrounds(self):
        cRound = random.randint(2,25)
        listInst = build_consistent_generator()(100)
        br = dtree.boost(listInst, cMaxRounds=cRound)
        self.assertTrue(len(br.listCfer) <= cRound)
        self.assertTrue(len(br.listDblCferWeight) <= cRound)

    @repeated
    def test_classify_boosted(self):
        def build_stump(fPolarity):
            dt = dtree.DTree(ixAttr=0,fDefaultLabel=True)
            dt.add(dtree.DTree(fLabel=fPolarity),0)
            dt.add(dtree.DTree(fLabel=not fPolarity),1)
            return dt
        cCfer = 10 
        listCfer = [build_stump(bool(i%2)) for i in xrange(cCfer)]
        listWeight = [math.exp(-i) for i in xrange(cCfer)]
        inst = dtree.Instance([int(randbool())], randbool())
        fLabel = dtree.classify_boosted(dtree.BoostResult(listWeight,listCfer),
                                        inst)
        self.assertEqual(bool(inst.listAttrs[0]), fLabel)

    @repeated
    def test_yield_boosted_folds(self):
        fxnCheck = lambda cvf: isinstance(cvf,dtree.BoostedFold)
        is_valid_cvf_builder(self, dtree.yield_boosted_folds, fxnCheck, False)
        
if __name__ == "__main__":
    import sys
    sys.exit(unittest.main())

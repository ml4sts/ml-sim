from mlsim import bias
from mlsim.bias.bias_components import Feature

def test_overall():
    assert(bias.Population())

def test_base_constructor():

    test_pop = bias.Population()

    # Makes sure the default parameters are set correctly
    assert test_pop.demographic_sampler.__dict__ == bias.bias_components.Demographic().__dict__
    assert test_pop.feature_sampler.__dict__ == bias.bias_components.Feature().__dict__
    assert test_pop.target_sampler.__dict__ == bias.bias_components.Target().__dict__
    assert test_pop.feature_noise_sampler.__dict__ == bias.bias_components.FeatureNoise().__dict__


def test_constructor_with_params():
    # Custom Parameters
    tempDic = {'dem':(.6,.4),
               'target':(0.06,),
               'feat':('mean_only_mvn',[[4,2],[2,4]]),
               'featnoise':('shape_spread_only_mvn',0.9)}
    testPopulation = bias.Population(parameter_dictionary=tempDic)

    # Assigns Custom parameters to individual components
    testDem = bias.Demographic(.6,.4)
    testTarget = bias.Target(0.06)
    testFeature = bias.Feature('mean_only_mvn',[[4,2],[2,4]])
    testFeatureNoise = bias.FeatureNoise('shape_spread_only_mvn',0.9)

    # Creates description of the parameters the same way Populations does
    description = ''
    description += 'Demographic Parameters\n'
    description += testDem.params.__str__()
    description += '\nTarget Parameters \n'
    description += testTarget.params.__str__()
    description += '\nFeature Parameters \n'
    description += testFeature.params.__str__()
    description += '\nFeature Noise Parameters \n'
    description += testFeatureNoise.params.__str__()

    # Compares parameters to make sure they are set correctly in Populations
    assert testPopulation.get_parameter_description() == description

def test_demographic_sampler():
    testPop = bias.Population()
    #Number of samples used in Population class
    sampleNum = 3000
    # How close to the target parameter it has to be
    accuracyThreshold = .03

    df = testPop.sample(sampleNum)
    probA = sum(df['a'])/sampleNum
    probZ = sum(df['z'])/sampleNum

    # Checks to make sure target probability and actual are close enough
    assert abs(probA - testPop.demographic_sampler.get_rho_a()) < accuracyThreshold
    assert abs(probZ - testPop.demographic_sampler.get_rho_z()[0]) < accuracyThreshold

def test_target_sampler():
    testPop = bias.Population()
    # Number of Samples used in Population class
    sampleNum = 4000
    # How close to the target parameter it has to be
    accuracyThreshold = .01
    df = testPop.sample(sampleNum)
    probY = sum(df['y'])/sampleNum
    probZ = sum(df['z'])/sampleNum
    # Checks to make sure target probability and actual are close enough
    assert abs(probY-probZ) < accuracyThreshold

def test_feature_sampler():
    N = 4000
    testFeat = bias.bias_components.Feature()
    a,z = bias.bias_components.Demographic().sample(N)
    y = bias.bias_components.Target().sample(a,z)
    x = bias.bias_components.Feature().sample(a,z,y)

    # TODO: assert X matches MU parameter (MU = [[5,2],[2,5]] by defualt)

#def test_feature_noise_sampler():
    # TODO: Check For Noise


def test_proxy_targets_constructor():
    # default construction works and is a Population subclass
    pop = bias.PopulationProxyTargets()
    assert isinstance(pop, bias.Population)
    # default noise sampler is set (matches parent convention, not None)
    assert pop.feature_noise_sampler.__dict__ == bias.FeatureNoise().__dict__


def test_proxy_targets_columns_and_shape():
    pop = bias.PopulationProxyTargets(
        feature_sampler_true=bias.FeatureTwoParams(loc=[[0, 0], [3, 3]],
                                                   spread=[1, 1]),
        feature_sampler_proxy=bias.FeatureTwoParams(loc=[[-2.5, -2.5], [2.5, 2.5]],
                                                    spread=[1, 1]),
    )
    df = pop.sample(500)
    # a, z, y + 2 truth features + 2 proxy features
    assert list(df.columns) == ['a', 'z', 'y', 'x0', 'x1', 'x2', 'x3']
    assert df.shape == (500, 7)


def test_proxy_targets_label_error_per_group():
    # the bias source: group 0 has much more label error (y != z) than group 1
    pop = bias.PopulationProxyTargets(
        demographic_sampler=bias.DemographicCorrelated(rho_a=0.5, rho_z=[0.5, 0.5]),
        target_sampler=bias.TargetTwoError(beta=[0.45, 0.05]),
    )
    df = pop.sample(6000)
    err0 = (df[df['a'] == 0]['y'] != df[df['a'] == 0]['z']).mean()
    err1 = (df[df['a'] == 1]['y'] != df[df['a'] == 1]['z']).mean()
    assert err0 > err1
    assert abs(err0 - 0.45) < 0.05
    assert abs(err1 - 0.05) < 0.03


def test_proxy_targets_accuracy_gap():
    # optional: needs scikit-learn (pulled in transitively by aif360)
    import pytest
    pytest.importorskip("sklearn")
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    pop = bias.PopulationProxyTargets(
        demographic_sampler=bias.DemographicCorrelated(rho_a=0.5, rho_z=[0.5, 0.5]),
        target_sampler=bias.TargetTwoError(beta=[0.45, 0.05]),
        feature_sampler_true=bias.FeatureTwoParams(loc=[[0, 0], [3, 3]], spread=[1, 1]),
        feature_sampler_proxy=bias.FeatureTwoParams(loc=[[-2.5, -2.5], [2.5, 2.5]],
                                                    spread=[1, 1]),
    )
    df = pop.sample(5000)
    feats = ['x0', 'x1', 'x2', 'x3']
    tr, te = train_test_split(df, test_size=0.3, random_state=0)
    pred = GaussianNB().fit(tr[feats], tr['y']).predict(te[feats])
    g0 = accuracy_score(te['z'][te['a'] == 0], pred[te['a'] == 0])
    g1 = accuracy_score(te['z'][te['a'] == 1], pred[te['a'] == 1])
    # a=0 (more label error) is worse on the true target than a=1
    assert g0 < g1

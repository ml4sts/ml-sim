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

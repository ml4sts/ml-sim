import mlsim
from mlsim.bias_components import Feature

def test_overall():
    assert(mlsim.Population())
    
def test_base_constructor():
    
    test_pop = mlsim.Population()
    
    # Makes sure the default parameters are set correctly
    assert test_pop.demographic_sampler.__dict__ == mlsim.bias_components.Demographic().__dict__
    assert test_pop.feature_sampler.__dict__ == mlsim.bias_components.Feature().__dict__
    assert test_pop.target_sampler.__dict__ == mlsim.bias_components.Target().__dict__
    assert test_pop.feature_noise_sampler.__dict__ == mlsim.bias_components.FeatureNoise().__dict__
    
    
def test_constructor_with_params():
    # Custom Parameters
    tempDic = {'dem':(.6,.4),
               'target':(0.06,),
               'feat':('mean_only_mvn',[[4,2],[2,4]]),
               'featnoise':('shape_spread_only_mvn',0.9)}
    testPopulation = mlsim.Population(parameter_dictionary=tempDic)
    
    # Assigns Custom parameters to individual components
    testDem = mlsim.Demographic(.6,.4)
    testTarget = mlsim.Target(0.06)
    testFeature = mlsim.Feature('mean_only_mvn',[[4,2],[2,4]])
    testFeatureNoise = mlsim.FeatureNoise('shape_spread_only_mvn',0.9)
    
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
    testPop = mlsim.Population()
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
    testPop = mlsim.Population()
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
    testFeat = mlsim.bias_components.Feature()
    a,z = mlsim.bias_components.Demographic().sample(N)
    y = mlsim.bias_components.Target().sample(a,z)
    x = mlsim.bias_components.Feature().sample(a,z,y)
    
    # TODO: assert X matches MU parameter (MU = [[5,2],[2,5]] by defualt)
    
#def test_feature_noise_sampler():
    # TODO: Check For Noise
    
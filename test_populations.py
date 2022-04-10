import mlsim
from mlsim.bias_components import Feature

def test_overall():
    assert(mlsim.Population())
    
def test_base_constructor():
    
    test_pop = mlsim.Population()
    assert test_pop.demographic_sampler.__dict__ == mlsim.bias_components.Demographic().__dict__
    assert test_pop.feature_sampler.__dict__ == mlsim.bias_components.Feature().__dict__
    assert test_pop.target_sampler.__dict__ == mlsim.bias_components.Target().__dict__
    assert test_pop.feature_noise_sampler.__dict__ == mlsim.bias_components.FeatureNoise().__dict__
    
    
def test_constructor_with_params():
    tempDic = {'dem':(.6,.4),
               'target':(0.06,),
               'feat':('mean_only_mvn',[[4,2],[2,4]]),
               'featnoise':('shape_spread_only_mvn',0.9)}
    testPopulation = mlsim.Population(parameter_dictionary=tempDic)
    
    testDem = mlsim.Demographic(.6,.4)
    testTarget = mlsim.Target(0.06)
    testFeature = mlsim.Feature('mean_only_mvn',[[4,2],[2,4]])
    testFeatureNoise = mlsim.FeatureNoise('shape_spread_only_mvn',0.9)
    
    description = ''
    description += 'Demographic Parameters\n'
    description += testDem.params.__str__()
    description += '\nTarget Parameters \n'
    description += testTarget.params.__str__()
    description += '\nFeature Parameters \n'
    description += testFeature.params.__str__()
    description += '\nFeature Noise Parameters \n'
    description += testFeatureNoise.params.__str__()
    
    
    assert testPopulation.get_parameter_description() == description
    
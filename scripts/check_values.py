import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score,plot_confusion_matrix,classification_report

vfs = np.loadtxt("TerminalValues1.txt")

rews = np.loadtxt("RewardsPerStep1.txt")

vehicle_info = pkl.load(open("vehicle_info1.pkl","rb"))

obs = pkl.load(open("observations1.pkl","rb"))

print("vehicle_info",len(vehicle_info))
print("rews",len(rews))
print("length of obs",len(obs))
print("length of vfs",len(vfs))

print("first observation",obs[0].shape)



rollouts = []
vehicle_info_rollouts = []
rollout_rewards = []
organized_obs = []


individual_rollout = []
vehicle_info_per_rollout = []
rew_per_rollout = []
per_rollout_obs = []

for i in range(len(vfs)):
	if vfs[i] == -100.0:

		rollouts.append(individual_rollout)
		vehicle_info_rollouts.append(vehicle_info_per_rollout)
		rollout_rewards.append(rew_per_rollout)
		organized_obs.append(per_rollout_obs)

		individual_rollout = []
		vehicle_info_per_rollout = []
		rew_per_rollout = []
		per_rollout_obs = []
	else:
		individual_rollout.append(vfs[i])
		vehicle_info_per_rollout.append(vehicle_info[i])
		rew_per_rollout.append(rews[i])
		per_rollout_obs.append(obs[i])


print("length of organized values",len(rollouts))
print("length of organized vehicle info",len(vehicle_info_rollouts))
print("length of organized rewards",len(rollout_rewards))
print("length of organized obs",len(organized_obs))
input()




if len(rollouts) == len(vehicle_info_rollouts):
	print(len(rollouts))
	print("CORRECT !!!")

rollout_lens = []

incomplete_rollouts = [i for i in range(len(rollouts)) if len(rollouts[i]) < 100]
full_rollouts = [i for i in range(len(rollouts)) if len(rollouts[i]) >= 100]

print("Number of rollouts that crashed",len(incomplete_rollouts))
print("Number of rollouts that completed successfully",len(full_rollouts))

input()

interesting_states = []
interesting_observations = []
for i in range(len(incomplete_rollouts)):
	if len(vehicle_info_rollouts[incomplete_rollouts[i]]) > 50:
		#interesting_states.append(vehicle_info_rollouts[incomplete_rollouts[i]][-50])
		#interesting_states.append(vehicle_info_rollouts[incomplete_rollouts[i]][-45])
		interesting_states.append(vehicle_info_rollouts[incomplete_rollouts[i]][-40])
		interesting_states.append(vehicle_info_rollouts[incomplete_rollouts[i]][-35])
		interesting_states.append(vehicle_info_rollouts[incomplete_rollouts[i]][-30])
		interesting_states.append(vehicle_info_rollouts[incomplete_rollouts[i]][-25])
		interesting_states.append(vehicle_info_rollouts[incomplete_rollouts[i]][-20])
		interesting_states.append(vehicle_info_rollouts[incomplete_rollouts[i]][-15])
		interesting_states.append(vehicle_info_rollouts[incomplete_rollouts[i]][-10])

		# store the observations
		#interesting_observations.append(organized_obs[incomplete_rollouts[i]][-50].flatten())
		#interesting_observations.append(organized_obs[incomplete_rollouts[i]][-45].flatten())
		#interesting_observations.append(organized_obs[incomplete_rollouts[i]][-40].flatten())
		for j in range(25):
			interesting_observations.append(organized_obs[incomplete_rollouts[i]][-j].flatten())
		#interesting_observations.append(organized_obs[incomplete_rollouts[i]][-10].flatten())
		#interesting_observations.append(organized_obs[incomplete_rollouts[i]][-5].flatten())



boring_observations = []

for i in range(len(interesting_observations)):
	random_rollout = np.random.randint(low=0,high=len(full_rollouts))
	random_index = np.random.randint(low=0,high=30)
	state = organized_obs[full_rollouts[random_rollout]][random_index]
	boring_observations.append(state.flatten())



label_interesting = [1]*len(interesting_observations)
label_boring = [0]*len(boring_observations)

data_x = interesting_observations + boring_observations
data_y = label_interesting + label_boring

print("length of data ",np.asarray(data_x).shape)
print("lenght of labels",np.asarray(data_y).shape)


print("length of interesting state distribution",len(interesting_observations))
print("length of boring state distribution",len(boring_observations))

input()


trainX,testX,trainY,testY = train_test_split(data_x,data_y,test_size=0.2)
print(len(trainX))


sc = StandardScaler()

scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)
print("TRAINING HAS STARTED ######################################################")



mlp_clf = MLPClassifier(hidden_layer_sizes=(1024,512,1024),
						learning_rate = 'constant',
						learning_rate_init=0.005,
						max_iter=5000,
						activation='relu',
						solver='adam')

KNN_model = KNeighborsClassifier(n_neighbors=3)



mlp_clf.fit(trainX,trainY)
KNN_model.fit(trainX,trainY)

y_pred = mlp_clf.predict(testX)
y_pred_knn = KNN_model.predict(testX)


print("metrics of nn model")
print('Accuracy: {:.2f}'.format(accuracy_score(testY,y_pred)))

print(classification_report(testY,y_pred))

print("metrics of knn model")
print('Accuracy: {:.2f}'.format(accuracy_score(testY,y_pred_knn)))

print(classification_report(testY,y_pred_knn))


#save both models
filename = "NNModel_node.sav"
pkl.dump(mlp_clf,open(filename,"wb"))

filename = "KNNModel_node.sav"
pkl.dump(KNN_model,open(filename,"wb"))

plt.plot(mlp_clf.loss_curve_)
plt.title("Loss Curve",fontsize=14)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

input()


pkl.dump(interesting_states,open("interesting_states_20_node0.pkl","wb"))


print("indices of incomplete rollouts",incomplete_rollouts)
print("percentage of incomplete rollouts",len(incomplete_rollouts)/1000)


for i in range(len(rollouts)):
	rollout_lens.append(len(rollouts[i]))

pdb.set_trace()

plt.plot(rollout_lens)
plt.show()



//	Andrew Blackledge
//	requires mnist datasets
//	runs several threads as built, expect it to max out your CPU on anything that isn't a workhorse
//	ouputs to several csv's one for each test

#include "stdafx.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>

double randomFloat() {
	return ((double)std::rand()) / ((double)RAND_MAX);
}

class Matrix {
public:
	int height;
	int width;
	float* location;

	Matrix(int x, int y) {
		width = x;
		height = y;
		location = (float*)std::malloc(sizeof(float)*height*width);
	}
	~Matrix() {
		free(location);
	}
	float& operator[] (const int index)
	{
		return location[index];
	}
	Matrix* Multiply(Matrix* partner, Matrix* result = nullptr) {

		if (partner->height != width) {
			return nullptr;
		}

		if (result == nullptr) {
			result = new Matrix(partner->width, height);
		}
		else if(result->height != height || partner->width != result->width){

			free(result->location);
			result->location = (float*)std::malloc(sizeof(float)*height*partner->width);

			result->height = height;
			result->width = partner->width;
		}

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < partner->width; x++) {
				float product = 0;

				//dot product time
				for (int i = 0; i < width; i++) {
					product += location[y*width + i] * partner->location[i*partner->width + x];
				}

				result->location[y*partner->width + x] = product;
			}
		}

		return result;
	}
	void Seed(float s) {
		if (s > 0) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					location[y*width + x] = 2 * (randomFloat() - .5)*s;
				}
			}
		}
		else {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					location[y*width + x] = 0;
				}
			}
		}
	}
};

class Layer {
protected:
	Layer* foreLayer;
	Matrix* weights;
	Matrix* momentum;
	int size;
public:
	Matrix* values;
	Matrix* deltas;

	Layer(int nodes, Layer* previous) {
		foreLayer = previous;
		size = nodes;
		
		values = new Matrix(1, size);
		if (previous != nullptr) {
			deltas = new Matrix(size, 1);
			weights = new Matrix(foreLayer->Size(), size);
			momentum = new Matrix(foreLayer->Size(), size);
		}
		else {
			deltas = nullptr;
			weights = nullptr;
			momentum = nullptr;
		}
	}
	void Seed(float s) {
		if (deltas != nullptr) {
			weights->Seed(s);
			momentum->Seed(0);
			foreLayer->Seed(s);
		}
	}
	int Size() {
		return size;
	}
	float Sigmoid(float z) {
		return std::pow(1 + std::pow(2.71828, -z), -1);
	}
	virtual void UpdateValues() {
		//recurses backwards across the network, then calculates layer values as it goes back
		foreLayer->UpdateValues();

		//calculate sums
		weights->Multiply(foreLayer->values, values);

		//apply sigmoid function to sums
		for (int i = 0; i < size; i++) {
			(*values)[i] = Sigmoid((*values)[i]);
		}
	}
	virtual void UpdateWeights(float LR, float MR) {
		if(foreLayer->deltas != nullptr){ //is this not the input layer?
			//update fore layer deltas, before updating this layers weights

			int psize = foreLayer->Size();

			deltas->Multiply(weights, foreLayer->deltas);

			for (int x = 0; x < psize; x++) {
				foreLayer->deltas->location[x] *= foreLayer->values->location[x] * (1 - foreLayer->values->location[x]);
			}
		}

		//update this layers weights
		int width = foreLayer->Size();
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < size; y++) {
				//calculate change and store it
				(*momentum)[y*width + x] = (*foreLayer->values)[x] * (*deltas)[y] * LR + (*momentum)[y*width+x] * MR;
				//apply it
				(*weights)[y*width + x] += (*momentum)[y*width + x];
			}
		}

		if (foreLayer->deltas != nullptr) {//is this not the input layer?
			//recurse back across net
			foreLayer->UpdateWeights(LR, MR);
		}
	}
};

class Input : public Layer {
public:
	Input(int nodes) : Layer(nodes, nullptr) {
	}
	void UpdateValues() {
		return; //override function since this layers values are handled by externally through setinput
	}
	void SetInput(Matrix* next) {
		values = next;
	}
};

class Output : public Layer {
public:
	Output(int nodes, Layer* previous) : Layer(nodes, previous) {

	}
	void UpdateWeights(float LR, float MR, int c) {
		//given a classification, calculate error costs and store them as deltas
		for (int v = 0; v < size; v++) {
			if (v == c) {
				(*deltas)[v] = (*values)[v] * (1 - (*values)[v])*(.9 - (*values)[v]);
			}
			else {
				(*deltas)[v] = (*values)[v] * (1 - (*values)[v])*(.1 - (*values)[v]);
			}
		}
		Layer::UpdateWeights(LR, MR);
	}
};

struct Image : public Matrix {
	//associates a class with an input vector
	int classification;

	Image(int size, int c) : Matrix(1, size + 1) {
		classification = c;
	}
};

struct ImageSet {
	//stores many images and provides a way of loading them
	std::vector<Image*> members;
	ImageSet(ImageSet* target, int size) {
		int psize = target->members.size();

		//create a shuffled index array
		int* indexSet = (int*)malloc(sizeof(int)*psize);
		for (int v = 0; v < psize; v++) {
			indexSet[v] = v;
		}

		std::random_shuffle(indexSet, indexSet + psize);

		for (int i = 0; i < size; i++) {
			members.push_back(target->members[indexSet[i]]);
		}
	}
	ImageSet(std::string fileName, int inputCount) {

		std::string heldLine;
		std::string::size_type tracker;
		std::ifstream file;

		file.open(fileName);
		if (!file.is_open()) {
			std::cout << "\nCould not open image set\n";
		}
		/*while(std::getline(file, heldLine)){
		Image* joiningMember = new Image(inputCount, std::stoi(heldLine, &tracker));

		for (int x = 0; x < inputCount; x++) {
		heldLine = heldLine.substr(++tracker);
		float value = std::stof(heldLine, &tracker) / 255.0;
		joiningMember->values->values[x] = value;
		}

		joiningMember->values->values[inputCount] = 1;

		members.push_back(joiningMember);
		}*/

		while (std::getline(file, heldLine, ',')) {
			Image* joiningMember = new Image(inputCount, std::stoi(heldLine));

			for (int x = 0; x < inputCount - 1; x++) {
				std::getline(file, heldLine, ',');
				float value = std::stof(heldLine) / 255;
				joiningMember->location[x] = value;
			}

			std::getline(file, heldLine);
			float value = std::stof(heldLine) / 255;
			joiningMember->location[inputCount - 1] = value;

			joiningMember->location[inputCount] = 1;

			members.push_back(joiningMember);


		}


	}
};

class NeuralNet {
	Input* Start;
	Output* End;
public:
	NeuralNet(int HL, int* LW) {

		Start = new Input(LW[0]);

		Layer* previous = (Layer*)Start;
		for (int v = 1; v <= HL; v++) {
			previous = new Layer(LW[v], previous);
		}

		End = new Output(LW[HL+1], previous);
	}
	void Seed(float s) {
		End->Seed(s);
	}
	void train(ImageSet* target, float LR, float MR) {
		//stochasticly trains all Perceptrons from an ImageSet
		int size = target->members.size();

		//create a shuffled index array
		int* indexSet = (int*)malloc(sizeof(int)*size);
		for (int v = 0; v < size; v++) {
			indexSet[v] = v;
		}

		std::random_shuffle(indexSet, indexSet + size);

		//iterate accross the index array, training in that order
		for (int v = 0; v < size; v++) {
			Image* next = target->members[indexSet[v]];
			Start->SetInput(next);

			End->UpdateValues();
			End->UpdateWeights(LR, MR, next->classification);
		}

	}
	int classify(Image* target) {
		//returns the most likely class of a single Image
		Start->SetInput(target);
		End->UpdateValues();

		float highestScore = 0;
		int scorer = -1;
		for (int v = 0; v < End->Size(); v++) {
			float score = (*End->values)[v];
			if (score > highestScore) {
				highestScore = score;
				scorer = v;
			}
		}
		return scorer;
	}
	float test(ImageSet* target) {
		//evaluates the accuraccy of the classifier for an ImageSet
		int correct = 0;

		for (int v = 0; v < target->members.size(); v++) {
			if (classify(target->members[v]) == target->members[v]->classification) {
				correct++;
			}
		}

		return (float)correct / (float)target->members.size();
	}
	Matrix* matrix(ImageSet* target) {
		Matrix* result =  new Matrix(10, 10);
		for (int v = 0; v < 100; v++) {
			(*result)[v] = 0;
		}
		for (int v = 0; v < target->members.size(); v++) {
			int predicted = classify(target->members[v]);
			int actual = target->members[v]->classification;
			(*result)[actual * 10 + predicted]++;
		}
		return result;
	}
};

float* TrainingCycle(NeuralNet* classifiers, ImageSet* trainer, ImageSet* tester, int epochCount, float LR, float MR, std::ofstream& output) {
	//stochasticly trains a classifier and stores the accuracy after every epoch
	classifiers->Seed(.5);

	float* accuracy = (float*)malloc(sizeof(float)*(epochCount + 1) * 2);
	for (int v = 0; v < epochCount; v++) {
		accuracy[v * 2] = classifiers->test(trainer);
		accuracy[v * 2 + 1] = classifiers->test(tester);

		output << accuracy[v * 2] << "," << accuracy[v * 2 + 1] << "\n";

		classifiers->train(trainer, LR, MR);
	}

	accuracy[epochCount * 2] = classifiers->test(trainer);
	accuracy[epochCount * 2 + 1] = classifiers->test(tester);

	output << accuracy[epochCount * 2] << "," << accuracy[epochCount * 2 + 1] << "\n";

	return accuracy;
}

void DisplayMatrix(Matrix* matrix, std::ofstream& output) {
	std::cout << "\n";
	for (int x = 0; x < 10; x++) {
		for (int y = 0; y < 10; y++) {
			std::cout << (*matrix)[x * 10 + y] << " ";
			output << (*matrix)[x * 10 + y] << ",";
		}
		std::cout << "\n";
		output << "\n";
	}
	std::cout << "\n";
	output << "\n";
	free(matrix);
}

void TestThread(ImageSet*trainSet, ImageSet*testSet, int H, double MR, int factor, std::ofstream  *output) {
	//thread based setup for a network, writes results to ouput, then closes the stream

	int large[3] = { 765, H, 10 };
	NeuralNet classifier(1, large);

	//this ends up looking pretty jumbled on account of the threads being launched at once... oh well
	std::cout << "--- START : "<<H<<" "<<MR<<" "<<1.0/factor<<" ---\n";
	if (factor > 1) {
		ImageSet partialTrainSet(trainSet, trainSet->members.size() / factor);

		TrainingCycle(&classifier, &partialTrainSet, testSet, 50, .1, MR, *output);
	}
	else {
		TrainingCycle(&classifier, trainSet, testSet, 50, .1, MR, *output);
	}
	DisplayMatrix(classifier.matrix(testSet), *output);
	std::cout << "--- END : " << H << " " << MR << " " << 1.0 / factor << " ---\n";

	output->close();
}

int main()
{
	//###########################
	//########## SETUP ##########
	//###########################

	int inputCount = 764;

	std::cout << "loading data...\n";
	ImageSet trainSet("mnist_train.csv", inputCount);
	ImageSet testSet("mnist_test.csv", inputCount);

	//open ALL the files
	std::ofstream hidden_a;
	hidden_a.open("hidden_twenty.csv");
	std::ofstream hidden_b;
	hidden_b.open("hidden_fifty.csv");
	std::ofstream hidden_c;
	hidden_c.open("hidden_hundred.csv");

	std::ofstream momentum_a;
	momentum_a.open("momentum_zero.csv");
	std::ofstream momentum_b;
	momentum_b.open("momentum_quarter.csv");
	std::ofstream momentum_c;
	momentum_c.open("momentum_half.csv");

	std::ofstream size_a;
	size_a.open("size_quarter.csv");
	std::ofstream size_b;
	size_b.open("size_half.csv");

	//launch a thread for each test
	std::thread threadHiddenA(TestThread, &trainSet, &testSet, 20, .9, 1, &hidden_a);
	std::thread threadHiddenB(TestThread, &trainSet, &testSet, 50, .9, 1, &hidden_b);
	std::thread threadHiddenC(TestThread, &trainSet, &testSet, 100, .9, 1, &hidden_c);

	std::thread threadMomentumA(TestThread, &trainSet, &testSet, 100, 0.0, 1, &momentum_a);
	std::thread threadMomentumB(TestThread, &trainSet, &testSet, 100, .25, 1, &momentum_b);
	std::thread threadMomentumC(TestThread, &trainSet, &testSet, 100, .5, 1, &momentum_c);

	std::thread threadSizeA(TestThread, &trainSet, &testSet, 100, .25, 4, &size_a);
	std::thread threadSizeB(TestThread, &trainSet, &testSet, 100, .5, 2, &size_b);
	
	//wait for it...
	threadHiddenA.join();
	threadHiddenB.join();
	threadHiddenC.join();

	threadMomentumA.join();
	threadMomentumB.join();
	threadMomentumC.join();

	threadSizeA.join();
	threadSizeB.join();

	//DONE
	std::cout << "\n----- DONE -----\n";
	std::cin >> inputCount;

	return 0;
}

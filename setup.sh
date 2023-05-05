#! /usr/bin/sh

# Create workspace
echo "Create a resource group:"
az group create --name "rg-dic" --location "eastus"

echo "Create an Azure Machine Learning workspace:"
az ml workspace create --name "mlw-dic" -g "rg-dic"

# Create compute instance

echo "Creating a compute instance with name: dic-compute" 
az ml compute create --name "dic-compute" --size STANDARD_DS11_V2 --type ComputeInstance -w mlw-dic -g rg-dic

# Create compute cluster
echo "Creating a compute cluster with name: aml-cluster"
az ml compute create --name "aml-cluster" --size STANDARD_DS11_V2 --max-instances 2 --type AmlCompute -w mlw-dic -g rg-dic
#! /usr/bin/sh

# Create resource group
echo "Create a resource group:"
az group create --name "rg-dic" --location "eastus"

# Create workspace
echo "Create an Azure Machine Learning workspace:"
az ml workspace create --name "mlw-dic" -g "rg-dic"

# Create compute instance
echo "Creating a compute instance with name: dic-compute" 
az ml compute create --name "dic-compute" --size STANDARD_DS11_V2 --type ComputeInstance -w mlw-dic -g rg-dic

# Create compute cluster
echo "Creating a compute cluster with name: aml-cluster"
az ml compute create --name "dic-cluster" --size Standard_D4s_v3 --min-instances 1 --max-instances 1 --type AmlCompute -w mlw-dic -g rg-dic
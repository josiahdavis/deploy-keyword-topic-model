name=$1
docker_image=${2:-216321755658.dkr.ecr.us-east-1.amazonaws.com/topic-modeling:latest}
iam_role=${3:-arn:aws:iam::216321755658:role/service-role/AmazonSageMaker-ExecutionRole-20171204T150334}

# Create model
aws sagemaker create-model \
    --model-name ${name} \
    --primary-container \
        Image=${docker_image} \
    --execution-role-arn ${iam_role}

# Create the endpoint configuration
aws sagemaker create-endpoint-config \
    --endpoint-config-name ${name} \
    --production-variants \
        VariantName=dev,ModelName=${name},InitialInstanceCount=1,InstanceType=ml.m4.xlarge,InitialVariantWeight=1.0

# Create endpoint
aws sagemaker create-endpoint \
    --endpoint-name ${name} \
    --endpoint-config-name ${name}

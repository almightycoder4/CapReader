FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt

# Install OpenCV dependencies
# RUN yum install -y libSM libXrender libXext

# Copy function code
COPY main.py ${LAMBDA_TASK_ROOT}/
COPY app ${LAMBDA_TASK_ROOT}/app/
COPY models ${LAMBDA_TASK_ROOT}/models/
COPY local_test.py ${LAMBDA_TASK_ROOT}/

# Ensure the model files have correct permissions
RUN chmod -R 755 ${LAMBDA_TASK_ROOT}/models/
RUN chmod -R 755 ${LAMBDA_TASK_ROOT}
RUN chmod -R 755 ${LAMBDA_TASK_ROOT}/

# Set the CMD to run the test script
CMD [ "main.lambda_handler" ]

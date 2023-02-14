import numpy as np
import tritonclient.grpc


def downcast_array(input_arr, target_type="np.float32"):
    """
    Downcast a numpy array input_arr from:
    - numpy.float64 to numpy.float32
    - numpy.float32 to python float.
    
    Note that in case of conversion to a flow 
    we also have to convert the numpy array 
    to a list to be able to use standard python types.
        
    """  
    
    if target_type == "float" or target_type is float:
        # conversion to python float requires
        # also converting array to a list
        return [el.item() for el in input_arr]
    
    if target_type == "np.float32" or target_type is np.float32:
        return input_arr.astype(np.float32)
    
    return input_arr


def triton_predict_grpc(triton_client, input_arr, model_name, batch_size=1, model_ver=1, datatype='FP32'):
        
    triton_input = tritonclient.grpc.InferInput('input__0', input_arr.shape, datatype=datatype)
    
    triton_input.set_data_from_numpy(input_arr)
    
    triton_output = tritonclient.grpc.InferRequestedOutput('output__0')
    
    response = triton_client.infer(model_name, 
                                   model_version=str(model_ver), 
                                   inputs=[triton_input], 
                                   outputs=[triton_output]
                                  )
        
    return response.as_numpy('output__0')


if __name__ == "__main__":
    
    # caution: if the connection to localhost does not work,
    # use container IPAddres obtained e.g. from:
    # docker inspect triton | grep IPAddress
    # triton_host = 'localhost'
    triton_host = '172.17.0.2'
    # xtriton_host = '172.17.0.3'

    # triton_port_http = 8000
    triton_port_grpc = 8001

    triton_client_grpc = tritonclient.grpc.InferenceServerClient(url=f'{triton_host}:{triton_port_grpc}')
    
    assert triton_client_grpc.is_server_ready()


    model_name='ks_clf_xgb_model_cpu'
    
    model_version=1
    
    features_num = 238
    batch_size = 1

    fake_test_data = np.array(np.random.uniform(0, 0, size=(batch_size, features_num)))

    assert fake_test_data.shape[1] == features_num

    fake_test_data_float32 = downcast_array(input_arr=fake_test_data, target_type=np.float32)

    assert fake_test_data_float32.dtype == np.float32

    print("\nPredictors input array - type, shape, contents:")
    print(fake_test_data_float32.dtype)
    print(fake_test_data_float32.shape)
    print(fake_test_data_float32)

    # generate predictions using GRPC endpoint
    triton_result = triton_predict_grpc(triton_client=triton_client_grpc, 
                                   input_arr=fake_test_data_float32, 
                                   model_name=model_name, 
                                   model_ver=model_version, 
                                   datatype="FP32"
                                   # datatype="FP64"
                                  )

    print("\nReturned model predictions array shape:")
    print(triton_result.shape)

    print("\nReturned model predictions array:")
    print(triton_result)

    print("\nReturned class 1 proba prediction:")
    print(triton_result[0, 1])

import torch
import torch_geometric #torch_geometric == 1.6.1
# import community
import community.community_louvain
import numpy as np
import networkx
import argparse
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
from Sign_OPT import *
from torch_geometric.utils import to_networkx, from_networkx
import torch_geometric.transforms as T
from Gin import GIN, SAG, GUNet
from time import time
import random
from collections import Counter



from google.colab import files
from IPython.display import clear_output
import matplotlib.pyplot as plt
from numpy.random import randn
from time import sleep

# this part is added:
#-----------------------------------------------------------------------
def my_mode(sample):
    c = Counter(sample)
    return [k for k, v in c.items() if v == c.most_common(1)[0][1]][0]




def inject_node(x, num_inject, initialization, connection, dataset):



    # print("Edge Attributes are:",x.edge_attr)

    node_feature_dim = x.x.shape[1]
    injected_feature = torch.zeros(node_feature_dim)
    num_nodes_before_injection = x.num_nodes

    if initialization == "zero":
        pass
    elif initialization == "one":
        injected_feature = torch.ones(node_feature_dim)
    elif initialization == "random":
        Gaussian_mean = torch.mean(x.x, dim=0)
        Gaussian_std = torch.std(x.x,dim=0)
        # injected_feature = torch.empty(node_feature_dim).normal_(mean=Gaussian_mean, std=Gaussian_std)
        injected_feature = torch.empty(node_feature_dim)
        injected_feature[0].normal_(mean=Gaussian_mean[0].item(), std=Gaussian_std[0].item())
        if(dataset != 'IMDB-BINARY'):
            injected_feature[1].normal_(mean=Gaussian_mean[1].item(), std=Gaussian_std[1].item())
        # print(injected_feature)
    elif initialization == "node_mean":
        injected_feature = torch.mean(x.x, dim=0)
    else:
        print(f"Unsupported Initialization method: {initialization}")
        exit()
    # inject new nodes into x
    x.x = torch.cat((x.x, torch.tensor([injected_feature.cpu().numpy() for i in range(num_inject)]).cuda()))
    x.num_nodes = len(x.x)





    # connect new nodes into x
    if(connection == "no_connection"):
        pass
    elif(connection == "random"):
        for i in range(num_inject):
            node_number = i+num_nodes_before_injection
            random_node = random.randint(0,num_nodes_before_injection-1)

            new_edge_back = torch.tensor([[node_number],[random_node]]).cuda()
            new_edge_front = torch.tensor([[random_node],[node_number]]).cuda()

            x.edge_index = torch.cat((new_edge_front, x.edge_index, new_edge_back),1)
            if(x.edge_attr is not None):
                x.edge_attr = torch.cat((x.edge_attr, torch.tensor([[0.,1.],[0.,1.]]).cuda()),0)
            
    elif(connection == "mode"):
        for i in range(num_inject):
            node_number = i+num_nodes_before_injection
            mode_node = my_mode(x.edge_index[0].tolist())

            new_edge_back = torch.tensor([[node_number],[mode_node]]).cuda()
            new_edge_front = torch.tensor([[mode_node],[node_number]]).cuda()

            x.edge_index = torch.cat((new_edge_front, x.edge_index, new_edge_back),1)

            if(x.edge_attr is not None):
                x.edge_attr = torch.cat((x.edge_attr, torch.tensor([[0.,1.],[0.,1.]]).cuda()),0)
    else:
        print(f"Unsupported Connection method: {connection}")
        exit()

    x.num_edges = x.num_edges + num_inject

    
    return x

 # n = to_networkx(x, node_attrs=x.x, edge_attrs=x.edge_attr, to_undirected=True)
 # n = to_networkx(x, to_undirected=True)
 # n.add_edge(node_number, mode_node)
 # x = from_networkx(n)


#-----------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch graph isomorphism network for graph classification')
    #these are parameters for attack model


    # this part is added:
    #-----------------------------------------------------------------------
    parser.add_argument('--initialization', type=str, default='zero')
    parser.add_argument('--injection_percentage', type=float, default='0')
    parser.add_argument('--injection_number', type=int, default='0')
    parser.add_argument('--connection', type=str, default='no_connection')

    
    #-----------------------------------------------------------------------

    parser.add_argument('--effective', type=int, default=1)
    parser.add_argument('--max_query', type=int, default=40000)
    parser.add_argument('--id', type= int, default=1)
    parser.add_argument('--search', type=int, default=1)
    #these are parameters for GIN model
    parser.add_argument('--dataset', type=str, default="IMDB-BINARY")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32, help='social dataset:64 bio dataset:32')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--model_path', type=str, default='./trained_model/')
    parser.add_argument('--model', type=str, default='GUN')
    args = parser.parse_args()
    return args

def distance(x_adv, x):
    adj_adv = nx.adjacency_matrix(to_networkx(x_adv, to_undirected=True))
    adj_x = nx.adjacency_matrix(to_networkx(x, to_undirected=True))
    return np.sum(np.abs(adj_adv-adj_x)) / 2
    
def count_edges(x_adv, x):
    adj_adv = nx.adjacency_matrix(to_networkx(x_adv, to_undirected=True)).todense().A
    adj_x = nx.adjacency_matrix(to_networkx(x, to_undirected=True)).todense().A
    difference = adj_adv - adj_x
    num_add = sum(sum(difference==1)) / 2
    num_delete = sum(sum(difference==-1)) / 2
    return num_add, num_delete

TUD = {'NCI1':0,'COIL-DEL':0,'IMDB-BINARY':1}

if __name__ == '__main__':

    args = get_args()
    dataset_name = args.dataset
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else torch.device("cpu"))
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    model_path = args.model_path
    model_name = args.model
    injection_number = args.injection_number
    injection_percentage = args.injection_percentage
    initialization = args.initialization
    connection = args.connection
    
    if dataset_name in TUD.keys():
        degree_as_attr = TUD[dataset_name]
    else:
        print('invalid dataset!')
        raise(ValueError)

    if degree_as_attr:
        dataset = TUDataset(root='./dataset',name=dataset_name,use_edge_attr='False', use_node_attr=True,
        pre_transform=T.Constant(1, True))
    else:
        dataset = TUDataset(root='./dataset',name=dataset_name,use_edge_attr='False',use_node_attr=True)
    
    index_path = './data_split/' + dataset_name + '_'
    with open(index_path+'test_index.txt', 'r') as f:
        test_index = eval(f.read())
    test_dataset = dataset[test_index]
    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes
    print('input dim: ', input_dim)
    print('output dim: ', output_dim)
    print("\n \n \n")
    if model_name=='SAG':
        model = SAG(5,input_dim,hidden_dim,output_dim,0.8,dropout).to(device)
        load_path = model_path + '{}_{}.pt'.format(dataset_name, model_name)
    elif model_name=='GIN':
        model = GIN(5,2,input_dim,hidden_dim,output_dim,dropout).to(device)
        load_path = model_path + '{}_{}.pt'.format(dataset_name, model_name)
        # load_path = model_path + '{}.pt'.format(dataset_name)
    elif model_name=='GUN':
        model = GUNet(input_dim,hidden_dim,output_dim,0.8,3,dropout).to(device)
        load_path = model_path + '{}_{}.pt'.format(dataset_name, model_name)
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()

    attacker = OPT_attack_sign_SGD(model, device, args.effective)
    num_test = len(test_dataset)
    perturbation = [] #perturbation for each poisoned graph
    perturbation_ratio = [] #perturbation ratio for each poisoned graph

    no_need_count = 0
    num_query = []    
    fail_count = 0
    distortion = []
    attack_time = []

    init_perturbation = [] #perturbation for each poisoned graph
    init_perturbation_ratio = [] #perturbation ratio for each poisoned graph
    init_num_query = []    
    init_distortion = []
    init_attack_time = []
    search_type = []

    detect_test_normal = []
    detect_test_advers = []

    # this part is added:
    #-----------------------------------------------------------------------
    num_success = 0
    num_success_via_injection = 0
    #-----------------------------------------------------------------------

    num_add_edge, num_delete_edge = [], []


    num_isolated_nodes = 0


    for i in range(380,400):

        

        # print('begin to attack instance {}'.format(i))
        x0 = test_dataset[i].to(device)

        num_nodes_before_injection = x0.num_nodes
        
        # print("x0 edge index is:",x0.edge_index)
        # print("x0 edge index is:",x0.edge_index.t().contiguous())
        
        # print("Does x0 have any isolated nodes? --",x0.has_isolated_nodes())
        # if(x0.has_isolated_nodes()):
        #     num_isolated_nodes += 1
        


        # this part is added:
        #-----------------------------------------------------------------------
        # print("\n \n \n \n \n")
        print("---------------------------instance",i,"basic info-----------------------------------")
        # print("x0 before node injection is:", x0)
        # print("the information in x0 is:", x0.x)

        G0 = to_networkx(x0, to_undirected=True)




        print("nodes before injection:",list(G0.nodes))
        print("edges before injection:",list(G0.edges))
        # print("the information in G0 is:", G0.nodes.data())

        print("x0 edge index is:",x0.edge_index)






        #-----------------------------------------------------------------------
        y0 = x0.y[0]
        y1 = model.predict(x0, device)
        #-----------------------------------------------------------------------


        if injection_number == 0 and injection_percentage == 0:
            pass
        # if num_inject not specified, then use percentage
        elif injection_number == 0 and injection_percentage != 0:
            x0 = inject_node(x0, initialization=initialization, num_inject=max(1, int(x0.num_nodes*args.injection_percentage)), connection = connection, dataset = dataset_name)
        elif injection_number != 0 and injection_percentage == 0:
            x0 = inject_node(x0, initialization=initialization, num_inject = injection_number, connection = connection, dataset = dataset_name)
        else:
            print("Cannot have mixed specifications of injection_number and injection_percentage!")
            exit()


        num_nodes_after_injection = x0.num_nodes

        # print("x0 after node injection is:", x0)
        # print("the information in x0 is:", x0.x)
        G1 = to_networkx(x0, to_undirected=True)
        print("nodes after injection:",list(G1.nodes))
        print("edges after injection:",list(G1.edges))
        # print("the information in G1 is:", G1.nodes.data())

        print("x0 edge index is:",x0.edge_index)
        
        #-----------------------------------------------------------------------


        y2 = model.predict(x0, device)


        # this part is added:
        #-----------------------------------------------------------------------

        print("-----------------------------------------------------------------------------------")
        print("Ground truth (y0):", y0.item())
        print("Prediction before node injection (y1):", y1.item())
        print("Prediction after node injection (y2):", y2.item())
        print("-----------------------------------------------------------------------------------")

        '''
        please note here that y0 is the ground truth label for graph x0, y1 is the model prediction for x0 without any add-ons,
        y2 is the model prediction for x0 when nodes are inserted but no edge purturbations are in place
        we commence edge attack only when y2 != y0
        '''
        num_nodes = x0.num_nodes
        space = num_nodes * (num_nodes - 1) / 2

        
        if(y0 == y1 and y0 != y2):
            num_success_via_injection += 1
            print(f"case {i} is successfully attacked via node injection but without edge purturbation")



        #-----------------------------------------------------------------------

        

        elif(y0 == y1 and y0 == y2):
            time_start = time()
            adv_x0, adv_y0, query, success, dis, init = attacker.attack_untargeted(x0, y0, query_limit=args.max_query)
            # this part is added:
            #-----------------------------------------------------------------------
            print("before adv attack, the model predicts:", y0.item())
            print("after adv attack, the model predicts:", adv_y0.item())
            if y0 != adv_y0:
                num_success += 1
                print(f"case {i} is successfully attacked via node injection and edge purturbation")
            else:
                print(f"case {i} both attack methods failed")
            #-----------------------------------------------------------------------
            time_end = time()
            init_num_query.append(init[2])
            num_query.append(query)
            init_attack_time.append(init[3])
            attack_time.append(time_end-time_start)
            if success:


                #for visualization
                #-----------------------------------------------------------------------

                G2 = to_networkx(adv_x0, to_undirected=True)


                color0 = ['cornflowerblue'] * num_nodes_before_injection 
                color1 = ['cornflowerblue'] * num_nodes_before_injection + ['darkorange'] * (num_nodes_after_injection-num_nodes_before_injection)

    
    

                graphs = [G0,G1,G2]
                fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(21,7))
                ax = axes.flatten()
                nx.draw_networkx(graphs[0], ax=ax[0], with_labels=True, font_weight='bold',node_color=color0)
                ax[0].set_axis_off()
                nx.draw_networkx(graphs[1], ax=ax[1], with_labels=True, font_weight='bold',node_color=color1)
                ax[1].set_axis_off()
                nx.draw_networkx(graphs[2], ax=ax[2], with_labels=True, font_weight='bold',node_color=color1)
                ax[2].set_axis_off()

                line1 = plt.Line2D((.37,.37),(.1,.9), color="k", linewidth=3)
                line2 = plt.Line2D((.66,.66),(.1,.9), color="k", linewidth=3)
                # line2 = plt.Line2D((.1,.9),(.1,.1), color="k", linewidth=3)
                fig.add_artist(line1)
                fig.add_artist(line2)


                # ax[0].title.set_text('Before Injection')
                # ax[1].title.set_text('After Injection')

                ax[0].set_title('Before Node Injection', fontsize=16)
                ax[1].set_title('After Node Injection', fontsize=16)
                ax[2].set_title('After Edge Purturbation', fontsize=16)


                fig.show()
                # plt.show()



                filepath = '/content/drive/My Drive/249/Project/CCS21_GNNattack_Node_injection/Yu_Zhou/test_multi_node_mode/graph' + str(i) + '.png' 
                print(filepath)
                fig.savefig(filepath)

                # filename = 'Graph' + str(i) + '.png'
                # plt.savefig(filename)
                # files.download(filename) 




        
                #-----------------------------------------------------------------------

                #process results in Stage 1
                init_perturb, init_dis, init_query, init_time, s_type = init
                init_ratio = init_perturb / space
                init_perturbation.append(init_perturb)
                init_distortion.append(init_dis)
                search_type.append(s_type)
                init_perturbation_ratio.append(init_ratio)

                #process results in Stage 2
                perturb = distance(adv_x0, x0)
                perturbation.append(perturb)
                perturbation_ratio.append(perturb/space)
                distortion.append(dis)
                
                add_edge, delete_edge = count_edges(adv_x0, x0)
                num_delete_edge.append(delete_edge)
                num_add_edge.append(add_edge)

                #test dataset for defense
                #x0.y = torch.tensor([0])
                #adv_x0.y = torch.tensor([1])
                adv_x0.y = x0.y
                detect_test_advers.append(adv_x0)
                detect_test_normal.append(x0)
            else:
                detect_test_advers.append(x0)
                detect_test_normal.append(x0)
                init_distortion.append(-1)
                init_perturbation.append(-1)
                init_perturbation_ratio.append(-1)
                search_type.append(-1)

                perturbation.append(-1)
                perturbation_ratio.append(-1)
                distortion.append(-1) 
        else:
            print('instance {} is wrongly classified from the start, No Need to Attack'.format(i))
            no_need_count += 1
            num_query.append(0)
            attack_time.append(0)
            perturbation.append(0)
            perturbation_ratio.append(0)
            distortion.append(0)
            
            init_perturbation.append(0)
            init_distortion.append(0)
            init_num_query.append(0)
            init_attack_time.append(0)
            search_type.append(0)
            init_perturbation_ratio.append(0)

    
        # this part is changed:
        # -----------------------------------------------------------------------
        print("-----------------------------------------------------------------------------------")
        print(f"attack loop: success in {num_success} out of {i+1} instances, with {no_need_count} instances no need to attack, and {num_success_via_injection} cases attacked with only node injection and no edge purturbation")
        # print('{} instances don\'t need to be attacked'.format(no_need_count))
        if (i+1 - no_need_count - num_success_via_injection)*100 != 0:
            success_ratio = num_success / (i+1 - no_need_count - num_success_via_injection)*100
            print("So far success ratio is:", success_ratio, "%")
        print("\n \n \n \n \n \n \n \n")
    
    

    
    print("--------------------------Experiment Summary--------------------------")
    Cases_Need_Attack = num_test - no_need_count
    Cases_Need_Purturbation = num_test - no_need_count - num_success_via_injection
    Purturbation_success_ratio = (num_success / Cases_Need_Purturbation) *100

    avg_perturbation = sum(perturbation) / num_success
    print("Sign-Opt: the Purturbation_success_ratio of black-box attack is {}/{} = {:.4f}".format(num_success,num_test - no_need_count - num_success_via_injection, Purturbation_success_ratio))
    print('Sign-Opt: the average perturbation is {:.4f}'.format(avg_perturbation))
    print('Sign-Opt: the average perturbation ratio is {:.4f}'.format(sum(perturbation_ratio) / num_success*100))
    print('Sign-Opt: the average query count is {:.4f}'.format(sum(num_query)/(num_test-no_need_count)))
    print('Sign-Opt: the average attacking time is {:.4f}'.format(sum(attack_time)/(num_test-no_need_count)))
    print('Sign-Opt: the average distortion is {:.4f}'.format(sum(distortion)/num_success))
    print('dataset: {}'.format(dataset_name))

    # -----------------------------------------------------------------------


    if args.search == 1 and args.effective == 1 and args.id ==1: 
        detect_test_path = './defense/'+dataset_name+'_'+model_name+'_Our_'
        torch.save(detect_test_normal, detect_test_path+'test_normal.pt')
        torch.save(detect_test_advers, detect_test_path+'test_advers.pt')
        print('test dataset for defense saved!')
  
    
    init_path = './out1/init_{}_{}_{}_{}_'.format(dataset_name, args.id, args.effective , args.search)
    with open(init_path+'search_type.txt', 'w') as f:
        f.write(str(search_type))
    with open(init_path+'P.txt', 'w') as f:
        f.write(str(init_perturbation))
    with open(init_path+'PR.txt', 'w') as f:
        f.write(str(init_perturbation_ratio))
    with open(init_path+'D.txt', 'w') as f:
        f.write(str(init_distortion))
    with open(init_path+'Q.txt', 'w') as f:
        f.write(str(init_num_query))
    with open(init_path+'T.txt', 'w') as f:
        f.write(str(init_attack_time))  
    
    
    our_path = './out1/our_{}_{}_{}_{}_'.format(dataset_name, args.id, args.effective , args.search)
    with open(our_path+'Q.txt', 'w') as f:
        f.write(str(num_query))
    with open(our_path+'T.txt', 'w') as f:
        f.write(str(attack_time))
    with open(our_path+'P.txt', 'w') as f:
        f.write(str(perturbation))
    with open(our_path+'PR.txt', 'w') as f:
        f.write(str(perturbation_ratio))
    with open(our_path+'D.txt', 'w') as f:
        f.write(str(distortion))
    with open(our_path+'ADD.txt', 'w') as f:
        f.write(str(num_delete_edge))
    with open(our_path+'DEL.txt', 'w') as f:
        f.write(str(num_add_edge))
            
    print("the numbers of deleted edges are:", num_delete_edge)
    print("the numbers od added edges are:", num_add_edge)
    print("the average number of deleted edges for %s: %d"%(dataset_name, float(sum(num_delete_edge)/len(num_delete_edge))))
    print("the average number of added edges for %s: %d"%(dataset_name, float(sum(num_add_edge)/len(num_add_edge))))
    '''
    out_path = './out/{}_Opt_{}.txt'.format(dataset_name, bound)  
    with open(out_path, 'w') as f:
        f.write('{} instances don\'t need to be attacked\n'.format(no_need_count))
        f.write('Sign-Opt fails to attack {} instance\n'.format(fail_count))
        f.write("Sign-Opt: the success rate of black-box attack is {}/{} = {:.4f}\n".format(success_count,num_test-no_need_count, success_ratio))
        f.write('Sign-Opt: the average perturbation is {:.4f}\n'.format(avg_perturbation))
        f.write('Sign-Opt: the average perturbation ratio is {:.4f}\n'.format(sum(perturbation_ratio) / success_count*100))
        f.write('Sign-Opt: the average query count is {:.4f}\n'.format(sum(num_query)/(num_test-no_need_count)))
        f.write('Sign-Opt: the average attacking time is {:.4f}\n'.format(sum(attack_time)/(num_test-no_need_count)))
        f.write('Sign-Opt: the average distortion is {:.4f}\n'.format(sum(distortion)/success_count))
        f.write('Sign-Opt: detail perturbation are: {}\n'.format(perturbation))
        f.write('Sign-Opt: detail perturbation ratio are: {}\n'.format(perturbation_ratio))
    '''
    # print("number of cases with isolated nodes is:",num_isolated_nodes)

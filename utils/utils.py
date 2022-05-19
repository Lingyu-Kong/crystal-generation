from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
import torch
calc=LennardJones(rc=500)

def explore_by_dft(action,steps=100):
    pos=action[:,0:3].tolist()
    pos=list(map(tuple, pos))
    atoms=Atoms('C'+str(len(pos)),positions=pos)
    atoms.calc=calc
    dyn=BFGS(atoms,logfile=None)
    dyn.run(fmax=0.0001,steps=steps)
    out_pos=torch.tensor(atoms.positions)
    out_energy=atoms.get_potential_energy()
    out_action=torch.cat([out_pos,action[:,3:]],dim=1)
    return out_action,max(-out_energy,-5.0)

def evaluate_action_by_ase(action):
    pos=action[:,0:3].tolist()
    pos=list(map(tuple, pos))
    atoms=Atoms('C'+str(len(pos)),positions=pos)
    atoms.calc=calc
    return max(-atoms.get_potential_energy(),-5.0)

def free_mass_centre(action):
    mass_centre=torch.mean(action[:,0:3],axis=0)
    free_mass_pos=action[:,0:3]-mass_centre
    free_mass_action=torch.cat([free_mass_pos,action[:,3:]],dim=1)
    return free_mass_action

def get_random_rotation(pos):
    random_quaternions = torch.randn(4).to(pos)
    random_quaternions = random_quaternions / random_quaternions.norm(dim=-1, keepdim=True)
    return torch.einsum("kj,ij->ki", pos, quaternion_to_rotation_matrix(random_quaternions))

def quaternion_to_rotation_matrix(quaternion):
    q0 = quaternion[0]
    q1 = quaternion[1]
    q2 = quaternion[2]
    q3 = quaternion[3]
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).reshape(3, 3)

def check_if_action_accepted(action):
    pos=action[:,0:3]
    bond_list=torch.zeros(2,(pos.shape[0]-1)*pos.shape[0]//2,dtype=torch.int64)
    ctr=0
    for i in range(pos.shape[0]):
        for j in range(i+1,pos.shape[0]):
            bond_list[0][ctr]=i
            bond_list[1][ctr]=j
            ctr+=1
    sent_pos=pos[bond_list[0]]
    recv_pos=pos[bond_list[1]]
    distance=torch.norm(sent_pos-recv_pos,dim=-1)
    flag=torch.all(distance>=1)
    return flag
    

#!/usr/bin/env python
# Adiabatic Projection Self Interaction Correction 
# Benjamin G. Janesko 
#
# These routines implement the adiabatic projection self-interaction correction in PySCF 
# They are restricted to a single n-electron, m-orbital active space 
# PZSIC orbitals are obtained from 
# 
import numpy
import re 
import scipy 
import math 
from pyscf import scf, mcscf, gto, dft
from pyscf.lo.edmiston import ER 
from pyscf.lo.ibo import ibo
from scipy import linalg 
from pyscf.gto import moleintor
from pyscf.scf import hf
from pyscf.dft import numint
from os.path import exists 
from os import listdir 
a2k = 627.5095 

# Extract out AO-basis (PA,PB) projected onto orbitals orbs
# and the corresponding exact exchange energies 
def ProjDM(mf,PA,PB,orbs,Sm):
  S = mf.get_ovlp()
  Q = numpy.einsum('mi,ni->mn',orbs,orbs) # <mu|a><a|nu> active orbitals a 
  PpA = numpy.dot(Q,numpy.dot(S,numpy.dot(PA,numpy.dot(S,Q))))
  PpB = numpy.dot(Q,numpy.dot(S,numpy.dot(PB,numpy.dot(S,Q))))
  NA = numpy.einsum('mn,nm->',PpA,S)
  NB = numpy.einsum('mn,nm->',PpB,S)
  KA = -0.5*mf.get_k(dm=PpA)
  KB = -0.5*mf.get_k(dm=PpB)
  ExA = numpy.einsum('mn,nm->',PpA,KA)
  ExB = numpy.einsum('mn,nm->',PpB,KB)
  print('Proj: ',NA,NB,ExA,ExB)
  return(PpA,PpB,ExA,ExB)

# Extract out AO-basis (PA,PB) for a slice of MOs in a HF calculation 
def SliceDM(mf,sl):
  mos = mf.mo_coeff[:,sl]
  S = mf.get_ovlp()
  occs = mf.mo_occ[sl]
  nocc = occs.shape[0]
  occa =numpy.ndarray(nocc)
  occb =numpy.ndarray(nocc)
  for i in range(nocc):
    occa[i] = 0.
    occb[i] = 0.
    if(occs[i]>0.00001 and occs[i]<1.00001):
      occa[i] = occs[i]
    if(occs[i]>1.00001):
      occa[i] = 1.
      occb[i] = occs[i]-1.
  PA = numpy.einsum('mi,ni->mn',mos,numpy.einsum('ni,i->ni',mos,occa))
  PB = numpy.einsum('mi,ni->mn',mos,numpy.einsum('ni,i->ni',mos,occb))
  KA = -0.5*mf.get_k(dm=PA)
  KB = -0.5*mf.get_k(dm=PB)
  ExA = numpy.einsum('mn,nm->',PA,KA)
  ExB = numpy.einsum('mn,nm->',PB,KB)
  NA = numpy.einsum('mn,nm->',PA,S)
  NB = numpy.einsum('mn,nm->',PB,S)
  print('Slice: ',NA,NB)
  return(PA,PB,ExA,ExB)

# PBE and SIC-PBE XC from integer-occupied alpha-spin orbitals moas and
# occupied beta-spin orbitals mobs. The returned quantity "ESIC" should be
# added to the PBE XC energy to get the SIC-PBE XC energy. 
# This implementation uses Edmiston-Ruedenberg localized orbitals, with the
# initial guess generated from intrinsic bonding orbitals 
def DoSIC(ni, mol, myhf, grids, moas,mobs, relativity=0, hermi=0,
           max_memory=2000, verbose=None):

  if(moas.shape[1]<2):
    #print("You have one alpha orbital!")
    moas2 = moas 
  else:
    moas3 = ibo(mol,moas,locmethod='IBO',exponent=4)
    esa = ER(mol,moas3)
    esa.kernel()
    moas2 = esa.mo_coeff
  if(mobs.shape[1]<2):
    #print("You have one beta orbital!")
    mobs2 = mobs 
  else:
    mobs3 = ibo(mol,mobs,locmethod='IBO',exponent=4)
    esb = ER(mol,mobs3)
    esb.kernel()
    mobs2 = esb.mo_coeff
  
  gama= numpy.einsum('ia,ja->ij',moas2,moas2)
  gamb= numpy.einsum('ia,ja->ij',mobs2,mobs2)
  gam0 = gama*0.0
  (n,E) = PBEXC(ni,mol,grids,gama,gamb)
  ESIC = 0 
  for i in range(moas2.shape[1]):
    gam1 = numpy.einsum('i,j->ij',moas2[:,i],moas2[:,i])
    (n1,E1) = PBEXC(ni,mol,grids,gam1,gam0)
    EJ =  numpy.einsum('ij,ji->',gam1,myhf.get_j(dm=gam1)/2)
    print('a-Orbital ',i,'occ',n1,'Hartree ',EJ,' XC ',E1,' SIE ',EJ+E1)
    ESIC -= E1 + EJ 
  for i in range(mobs2.shape[1]):
    gam1 = numpy.einsum('i,j->ij',mobs2[:,i],mobs2[:,i])
    (n1,E1) = PBEXC(ni,mol,grids,gam1,gam0)
    EJ =  numpy.einsum('ij,ji->',gam1,myhf.get_j(dm=gam1)/2)
    print('b-Orbital ',i,'occ',n1,'Hartree ',EJ,' XC ',E1,' SIE ',EJ+E1)
    ESIC -= E1 + EJ 
  return(E,ESIC)

# PBE XC from PA,PB
def PBEXC(ni, mol, grids, dmas,dmbs, relativity=0, hermi=0,
           max_memory=2000, verbose=None):
    make_rhoa, nset, nao = ni._gen_rho_evaluator(mol, dmas, hermi)
    make_rhob, nset, nao = ni._gen_rho_evaluator(mol, dmbs, hermi)

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    aow = None
    ao_deriv = 1 
    for aos, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ao = aos[0]
        aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
        nao = ao.shape[0]
        for idm in range(nset):
            rhoas = make_rhoa(idm, aos, mask, 'GGA')
            rhobs = make_rhob(idm, aos, mask, 'GGA')
            exc,vxc = ni.eval_xc('PBE,PBE',(rhoas,rhobs),spin=1,relativity=relativity,deriv=1)[:2]
            den = (rhoas[0]+rhobs[0]) * weight
            nelec[idm] += den.sum()
            excsum[idm] += numpy.dot(den, exc)
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
    return nelec,excsum


def CAStest(m,nelec,norb,sortmos):
  # Compute APSIC energies from CASSCF orbitals 

  # RHF or ROHF calculation 
  s = m.spin
  if(s<1):
    doB = True  # doB is False if there are no minority-spin electrons 
    myhf=scf.RHF(m).newton()
    myhf.kernel()
  else:
    doB = False 
    if(m.nelec[1]>0): 
      doB = True
    myhf=scf.RHF(m).newton()
    myhf.kernel()# PySCF uses ROHF orbitals for open shell CASSCF 

  # CAS 
  if(norb<1):
    mycas = myhf
    ncas = 0 
    ncore = m.nelec[0]
  else:
    mycas=myhf.CASSCF(norb,nelec)
    mycas.fix_spin(s)
    if(len(sortmos)>0):
      mos = mycas.sort_mo(sortmos)
      mycas.run(mos)
    else:
      mycas.run() 
    ncas = mycas.ncas
    ncore = mycas.ncore
  
  # NEVPT2 reference calculation 
  EC = 0 
  if(norb>0):
    EC=mrpt.NEVPT(mycas,root=0).kernel()
  else:
    mp = myhf.MP2().run() 
    EC = mp.e_corr

  # Run APSIC 
  actsl = slice(ncore,ncore+ncas)
  actcsl = slice(0,ncore+ncas)
  csl = slice(0,ncore)
  return CorrTest(m,myhf,mycas,actsl,actcsl,csl,EC,1)

def CorrTest(m,myhf,myc,actsl,actcsl,csl,EC,typ):
  # Call with molecule, RHF, CAS/CC, slices of active and active+core and core spaces,
  # any extra EC to add, and the correlation type 
  # typ=1: CAS
  # typ=2: CC

  # Operators
  S=myhf.get_ovlp()
  Sm = linalg.inv(S)
  hcore = myhf.get_hcore()

  # Set up numerical integration for projected terms 
  df=dft.UKS(m)
  df.grids.level=4
  ni = df._numint

  # Full density matrices PA,PB and active space projections mosa,mosb
  (na,nb) = m.nelec
  mosa = myhf.mo_coeff[:,actsl]
  mosca = myhf.mo_coeff[:,actcsl]
  mosb = mosa 
  moscb = mosca 
  doConvert = False 
  if(typ==1):
    mosa = myc.mo_coeff[:,actsl]
    mosb = mosa
    mosca = myc.mo_coeff[:,actcsl]
    moscb = mosca
  s = m.spin
  if(s<1):
    if(typ==2):
      doConvert = True 
    PA=myc.make_rdm1()/2. 
    PAH=myhf.make_rdm1()/2. 
    PB = PA 
    PBH = PAH
  else:
    (PAH,PBH)=myhf.make_rdm1()
    if(typ==1):
      (PA,PB)=myc.analyze()
    else:
      if(myc.e_corr>-0.000001):
        (PA,PB) = (PAH,PBH)
      else:
        doConvert = True 
        (PA,PB)=myc.make_rdm1() # make_rdm1 is ill behaved where nvirt=0 
  if(doConvert):
    # MO to AO conversion 
    PA2 = PA 
    PA = numpy.einsum('mi,in->mn',myhf.mo_coeff,numpy.einsum('ij,mj->im',PA2,myhf.mo_coeff))
    PB2 = PB 
    PB = numpy.einsum('mi,in->mn',myhf.mo_coeff,numpy.einsum('ij,mj->im',PB2,myhf.mo_coeff))
  na = numpy.einsum('ij,ji->',PA,S)
  nb = numpy.einsum('ij,ji->',PB,S)
  nah = numpy.einsum('ij,ji->',PAH,S)
  nbh = numpy.einsum('ij,ji->',PBH,S)

  # Hartree+onelec and exchange energy at HF and CAS density 
  EH0 = numpy.einsum('ij,ji->',PA+PB,hcore+myhf.get_j(dm=PA+PB)/2)+myhf.energy_nuc()
  E0 =  myc.e_tot
  EH0H = numpy.einsum('ij,ji->',PAH+PBH,hcore+myhf.get_j(dm=PAH+PBH)/2)+myhf.energy_nuc()
  #print('Corr E',E0)
  #print('CorrPT2 E',E0+EC)
  #print('Corr Hartree+onelecE',EH0)
  #print('Corr EXC',E0-EH0)
  E0H =  myhf.e_tot

  EXH  = -numpy.einsum('ij,ji->',PAH,0.5*myhf.get_k(dm=PAH))
  EXH += -numpy.einsum('ij,ji->',PBH,0.5*myhf.get_k(dm=PBH))
  EX   = -numpy.einsum('ij,ji->',PA ,0.5*myhf.get_k(dm=PA ))
  EX  += -numpy.einsum('ij,ji->',PB ,0.5*myhf.get_k(dm=PB ))

  # Project out active density matrices and exchange contribution
  (PactA,PactB,EXactA,EXactB) = ProjDM(myhf,PA,PB,mosa,Sm)
  EXact = EXactA + EXactB
  #print('EXActA ',EXactA)
  EXc = EX  - EXactA - EXactB
  #print('Corr EX core',EXc)

  # Full and active PBEXC
  (NPBEH,EXCPBEH) = PBEXC(ni,m,df.grids,PAH,PBH)
  (NPBE,EXCPBE) = PBEXC(ni,m,df.grids,PA,PB)
  (NactPBE,EXCactPBE) = PBEXC(ni,m,df.grids,PactA,PactB)

  #print('PBE Ntot,Nact',NPBE,NactPBE)
  #print('PBE EXCtot,EXCact',EXCPBE,EXCactPBE)
  EXCcPBE = EXCPBE - EXCactPBE 
  EXCcPBEHH = 0.5*(EXCPBE - EXCactPBE  + EX-EXact)
  ECASPBE = E0-EXc + EXCcPBE 
  ECASPBEHH = E0-EXc + EXCcPBEHH

  # Full and core SIC
  (EXCPBEH2,SICPBEH) = DoSIC(ni,m,myhf,df.grids,myhf.mo_coeff[:,myhf.mo_occ>0],myhf.mo_coeff[:,myhf.mo_occ>1])
  EXCSICPBEH = EXCPBEH2 + SICPBEH 
  mocore = myc.mo_coeff[:,csl]
  (EXCcPBE2,SICcPBE) = DoSIC(ni,m,myhf,df.grids,mocore,mocore)
  ECASSICPBE = E0-EXc + EXCPBE - EXCactPBE + SICcPBE 

  #print("Tot,Tot,SIC post-HF PBE",EXCPBEH,EXCPBEH2,SICPBEH)
  #print("Tot,Tot,SIC core PBE",EXCcPBE,EXCcPBE2,SICcPBE)

  # Assemble the results: CASSCF,CASPT2, HF, PBE@HF, SICPBE@HF, AP-CAS-PBE, AP-SIC-PBE, AP-CAS-PBEHandH
  return(numpy.array((E0,E0+EC,E0H,EH0H+EXCPBEH,EH0H+EXCSICPBEH,ECASPBE,ECASSICPBE,ECASPBEHH)))

# Stretched N2 molecule , CAS(6,6)SCF reference 
def N2st(): 
  name = "N2st-cas66.txt" 
  f = open(name,"w")
  h=gto.Mole(atom='N',spin=3,basis='def2tzvp')#,ecp='def2tzvp')
  h.cart=True
  h.build()
  valsh  = CAStest(h,3,3,[])
  for r in (0.9,1.0,1.1,1.2,1.5,1.7,2.0,2.5,3.0,4.,5.,6.,8.,10.):
    print("\n\nN2 at ",r,"\n")
    st="N 0.0 0.0 0.0;N 0.0 0.0 %7.2f" % r
    h2=gto.Mole(atom=st,spin=0,charge=0,basis='def2tzvp')#,ecp='def2tzvp')
    h2.cart=True
    h2.build()
    vals = CAStest(h2,6,6,[]) 
    DE = 627.5095*(vals-2*valsh)
    str = " %7.2f   %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f \n" % (r,DE[0],DE[1],DE[2],DE[3],DE[4],DE[5],DE[6],DE[7])
    f.write(str)

# Stretched Li2(+) 
def Li2pst(): 
  name = "Li2pst.txt" 
  b = 'def2tzvp'
  f = open(name,"w")
  hp=gto.Mole(atom='Li',charge=1,spin=0,basis=b,cart=True)
  hp.build()
  valshp  = CAStest(hp,0,0,[])
  h=gto.Mole(atom='Li',spin=1,basis=b,cart=True)
  h.build()
  valsh  = CAStest(h,1,1,[])
  for r in (1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.8,4.0,5.0,6.0,7.0,8.,9.,10.,15.,20.0):
    print("\n\nLi2p at ",r,"\n")
    st="Li 0.0 0.0 0.0;Li 0.0 0.0 %7.2f" % r
    h2=gto.Mole(atom=st,spin=1,charge=1,cart=True,basis=b)
    h2.build()
    vals = CAStest(h2,1,2,[]) 
    DE = 627.5095*(vals-valsh-valshp)
    str = " %7.2f   %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f \n" % (r,DE[0],DE[1],DE[2],DE[3],DE[4],DE[5],DE[6],DE[7])
    f.write(str)
  
# Stretched Li2, CAS(2,2)SCF reference 
def Li2st(): 
  name = "Li2st.txt" 
  b = 'def2tzvp'
  f = open(name,"w")
  h=gto.Mole(atom='Li',spin=1,basis=b)
  h.cart=True
  h.build()
  valsh  = CAStest(h,1,1,[])
  for r in (1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.8,4.0,5.0,6.0,7.0,8.,9.,10.,15.,20.0):
    print("\n\nLi2 at ",r,"\n")
    st="Li 0.0 0.0 0.0;Li 0.0 0.0 %7.2f" % r
    h2=gto.Mole(atom=st,spin=0,charge=0,basis=b)
    h2.cart=True
    h2.build()
    vals = CAStest(h2,2,2,[]) 
    DE = 627.5095*(vals-2*valsh)
    str = " %7.2f   %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f \n" % (r,DE[0],DE[1],DE[2],DE[3],DE[4],DE[5],DE[6],DE[7])
    f.write(str)

# 2-butene torsional PES 
def twist():
  o = open("twist.txt","w")
  b = '6-31g'
  f0 = get_mol("./butene/butene-000.0.com",b)
  v0 = CAStest(f0,2,2,[])
  for fn in listdir("./butene"):
  #for fn in range(1):
    #tehfile= "./butene/butene-000.0.com" 
    #name = '000'
    tehfile= "./butene/%s" % fn
    name = fn.replace("butene-","").replace(".com","")
    print("\n\nNow doing ",name,"\n") 
    f= get_mol(tehfile,b)
    v = CAStest(f,2,2,[])
    DE = 627.5095*(v-v0)
    str = " %10s   %7.2f %7.2f %7.2f %7.2f    %7.2f %7.2f %7.2f %7.2f \n" % (name,DE[0],DE[1],DE[2],DE[3],DE[4],DE[5],DE[6],DE[7])  
    print(str) 
    o.write(str)

def ats0(name,q,sp,b):
  m=gto.Mole(atom=name,charge=q,spin=sp,basis=b)
  m.cart=True 
  m.build()
  mh=scf.RHF(m)
  mh.kernel() 
  df=dft.UKS(m)
  df.grids.level=4
  ni = df._numint
  print('Molecule ',name,' occs ',mh.mo_occ)
  mosa=mh.mo_coeff[:,mh.mo_occ>0]
  mosb=mh.mo_coeff[:,mh.mo_occ>1]
  (E,ESIC) = DoSIC(ni,m,mh,df.grids,mosa,mosb)
  return(ESIC) # We return the typically-negative value needed to correct the DFT energy. 


def ats():
  f = open("ats-sic.txt","w")
  b='6-311+G(3df,2p)'
  #b='sto-3g'
  v=ats0('H',0,1,b)
  f.write("H %8.4f\n"%v)
  #v=ats0('H',-1,0,b)
  #f.write("Hm %8.4f\n"%v)
  #v=ats0('He',1,1,b)
  #f.write("Hep %8.4f\n"%v)
  v=ats0('He',0,0,b)
  f.write("He %8.4f\n"%v)
  #v=ats0('Li',1,0,b)
  #f.write("Lip %8.4f\n"%v)
  v=ats0('Li',0,1,b)
  f.write("Li %8.4f\n"%v)
  #v=ats0('Li',-1,0,b)
  #f.write("Lim %8.4f\n"%v)
  v=ats0('Be',0,0,b)
  f.write("Be  %8.4f\n"%v)
  v=ats0('B',0,1,b)
  f.write("B   %8.4f\n"%v)
  v=ats0('C',0,2,b)
  f.write("C   %8.4f\n"%v)
  v=ats0('N',0,3,b)
  f.write("N   %8.4f\n"%v)
  v=ats0('O',0,2,b)
  f.write("O   %8.4f\n"%v)
  v=ats0('F',0,1,b)
  f.write("F   %8.4f\n"%v)
  v=ats0('Ne',0,0,b)
  f.write("Ne  %8.4f\n"%v)
  v=ats0('Na',0,1,b)
  f.write("Na  %8.4f\n"%v)
  v=ats0('Mg',0,0,b)
  f.write("Mg  %8.4f\n"%v)
  v=ats0('Al',0,1,b)
  f.write("Al  %8.4f\n"%v)
  v=ats0('Si',0,2,b)
  f.write("Si  %8.4f\n"%v)
  v=ats0('P',0,3,b)
  f.write("P  %8.4f\n"%v)
  v=ats0('S',0,2,b)
  f.write("S  %8.4f\n"%v)
  v=ats0('Cl',0,1,b)
  f.write("Cl  %8.4f\n"%v)
  v=ats0('Ar',0,0,b)
  f.write("Ar  %8.4f\n"%v)

def get_mol(fi,b):
  # Read Gaussian input file 
  ret = gto.Mole() 
  ret.cart=True 
  st=''
  q=0
  m=0
  lines=[] 
  with open(fi) as f:
    lines=f.readlines() 
  keep=0
  for line in lines:
    if(keep == 3 and len(line)>3): 
      ll = line.rstrip() 
      ll = re.sub('.fragment=..',' ',ll)
      ret.atom.append(ll)
    if(keep == 2):
      vals = line.split()  
      q=vals[0]
      m=vals[1]
      ret.charge=int(q)
      ret.spin=int(m)-1
      #print(q,m)
      keep=keep+1
    if re.match("^ *$",line)or re.match("^$",line):
      keep=keep+1
      #print(keep)
  ret.basis=b
  ret.build()
  return(ret) 
   

#!/usr/bin/perl

#read from training files

$CNFHOME="/home/zywang/work/raptorx-ss8-src";
require "$CNFHOME/bin/BioDSSP.pm";

(@ARGV >=2 ) or print <<"VEND";
Usage: GenFeat.train.pl [list files] [the output file]
Each line in the list file contiains.
[sequence file] [pssm file] [dssp file]

VEND

   open FH,"<$ARGV[0]" or die $!;
my @files=<FH>;
@files=map{my @p=split/\s+/;\@p}@files;
#die join(" ",@{$files[0]});
close FH;

open OFEAT,">$ARGV[1]";
open MATFILE,"<$CNFHOME/data/p.p2.c.unit"; #  

# Read feature matrix
my $i=0;
my $resorder="ARNDCQEGHILKMFPSTWYV";
my %resfeat;
my %aaorder=qw(A 0 R 1 N 2 D 3 C 4 Q 5 E 6 G 7 H 8 I 9 L 10  K 11 M 12  F 13  P 14 S 15  T 16  W 17  Y 18  V 19);
while(<MATFILE>)
{
    chomp;
    $_=~s/^\s+//;
    @parts=split/\s+/;
    $resfeat{substr($resorder,$i,1)}=$_;
    $i++;
    $resfeat{"X"}="0 "x scalar(@parts); 

}
close MATFILE;

print OFEAT scalar(@files),"\n";

for $xx(@files) { # sequence file, pssm file, dssp file
    my @ARGV=@{$xx};# die join(" ",@ARGV);
    my $dsspfile=$ARGV[2];
    my @dssp=getdssp($dsspfile);# die scalar(@{$dssp[1]});
    my @ss8=PSSConvertL8N8(@{$dssp[1]});
    my $seq;
    open fhSEQ,"<$ARGV[0]";
    $seq=<fhSEQ>;
    chomp($seq);
## End of reading
    $nsample=0;
    
#Parse PSSM
    my @tmprst=ParsePSSM("$ARGV[1]");
    my @pssmSeq=@{$tmprst[0]};
    my @pssmFeat=@{$tmprst[1]};
    
    my $pssmSequence=join "", @pssmSeq;
    die "$ARGV[0] PSSM contains gap or sequence inconsistent with the input sequence!\n$seq\n$pssmSequence\n" if ($seq ne $pssmSequence);
#add identity matrix
    my @allfeats;
    for($j=0;$j<scalar(@pssmSeq);$j++)
    {
	$allfeats[$j]=$pssmFeat[$j]." ".$resfeat{$pssmSeq[$j]};
    }
    
#Write all features to output file
#print OFEAT "1\n";
    print OFEAT scalar(@allfeats),"\n";
    print OFEAT join("\n",@allfeats);
    print OFEAT "\n";
#print OFEAT "0\n"x scalar(@pssmSeq);# all label are set to zeros
    die "$ARGV[0] sequence differ from label\n" if (scalar(@allfeats) !=scalar(@ss8));
    print OFEAT join("\n",@ss8),"\n";
    
    close fhSEQ;
}
close OFEAT;

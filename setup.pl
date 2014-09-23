#!/usr/bin/perl -w

#get perl executive path
$perlPath=`which perl`;
chomp($perlPath);
die "No perl found in searching folders.\n" if ($perlPath=~/no perl/);
die "setup.pl -home [raptorx-ss8 home] -blast [blast execute binary] -nr [nr database] " if @ARGV<3;
%params=@ARGV;

setsh("bin/run-train.sh");
setperl("bin/GenFeat.train.pl");

open fhRUN,"<".$params{"-home"}."/bin/run_raptorx-ss8.pl";
open fhRUNNEW,">".$params{"-home"}."/bin/run_cnfsseight.pl.new";
<fhRUN>;
print fhRUNNEW "#!$perlPath\n";
while(<fhRUN>){
    if(/^\$CNFHOME=/){
	print fhRUNNEW "\$CNFHOME=\"".$params{"-home"}."\";\n";
	next;
    }
    if(/^\$PSIBLASTEXE=/){
	print fhRUNNEW "\$PSIBLASTEXE=\"".$params{"-blast"}."\";\n";next;
    }
    if(/^\$NR=/){
	print fhRUNNEW "\$NR=\"".$params{"-nr"}."\";\n";next;
    }
    print fhRUNNEW "$_";

}

close fhRUN;
close fhRUNNEW;
$cmd="mv ".$params{"-home"}."/bin/run_cnfsseight.pl.new ".$params{"-home"}."/bin/run_raptorx-ss8.pl";
`$cmd`;
chmod(0755, $params{"-home"}."/bin/run_raptorx-ss8.pl");


open fhRUN,"<".$params{"-home"}."/bin/run_raptorx-ss3.pl";
open fhRUNNEW,">".$params{"-home"}."/bin/run_cnfsseight.pl.new";
<fhRUN>;
print fhRUNNEW "#!$perlPath\n";
while(<fhRUN>){
    if(/^\$CNFHOME=/){
	print fhRUNNEW "\$CNFHOME=\"".$params{"-home"}."\";\n";
	next;
    }
    if(/^\$PSIBLASTEXE=/){
	print fhRUNNEW "\$PSIBLASTEXE=\"".$params{"-blast"}."\";\n";next;
    }
    if(/^\$NR=/){
	print fhRUNNEW "\$NR=\"".$params{"-nr"}."\";\n";next;
    }
    print fhRUNNEW "$_";

}

close fhRUN;
close fhRUNNEW;
$cmd="mv ".$params{"-home"}."/bin/run_cnfsseight.pl.new ".$params{"-home"}."/bin/run_raptorx-ss3.pl";
`$cmd`;
chmod(0755, $params{"-home"}."/bin/run_raptorx-ss3.pl");


#----

open fhRUN,"<".$params{"-home"}."/bin/selftest.pl";
open fhRUNNEW,">".$params{"-home"}."/bin/run_cnfsseight.pl.new";
<fhRUN>;
print fhRUNNEW "#!$perlPath\n";
while(<fhRUN>){
    if(/^\$CNFHOME=/){
	print fhRUNNEW "\$CNFHOME=\"".$params{"-home"}."\";\n";
	next;
    }
    if(/^\$PSIBLASTEXE=/){
	print fhRUNNEW "\$PSIBLASTEXE=\"".$params{"-blast"}."\";\n";next;
    }
    if(/^\$NR=/){
	print fhRUNNEW "\$NR=\"".$params{"-nr"}."\";\n";next;
    }
    print fhRUNNEW "$_";

}

close fhRUN;
close fhRUNNEW;
$cmd="mv ".$params{"-home"}."/bin/run_cnfsseight.pl.new ".$params{"-home"}."/bin/selftest.pl";
`$cmd`;
chmod(0755, $params{"-home"}."/bin/selftest.pl");

sub setsh{

    my $fn=shift;
open fhRUN,"<".$params{"-home"}."/$fn";
open fhRUNNEW,">".$params{"-home"}."/$fn.new";
#<fhRUN>;
while(<fhRUN>){
    if(/^CNFHOME=/){
	print fhRUNNEW "CNFHOME=\"".$params{"-home"}."\";\n";
	next;
    }
    if(/^PSIBLASTEXE=/){
	print fhRUNNEW "PSIBLASTEXE=\"".$params{"-blast"}."\";\n";next;
    }
    if(/^NR=/){
	print fhRUNNEW "NR=\"".$params{"-nr"}."\";\n";next;
    }
    print fhRUNNEW "$_";
}

close fhRUN;
close fhRUNNEW;
$cmd="mv ".$params{"-home"}."/$fn.new ".$params{"-home"}."/$fn";
`$cmd`;
chmod(0755, $params{"-home"}."/$fn");

}

sub setperl{

    my $fn=shift;
open fhRUN,"<".$params{"-home"}."/$fn";
open fhRUNNEW,">".$params{"-home"}."/$fn.new";
<fhRUN>;
print fhRUNNEW "#!$perlPath\n";
while(<fhRUN>){
    if(/^\$CNFHOME=/){
	print fhRUNNEW "\$CNFHOME=\"".$params{"-home"}."\";\n";
	next;
    }
    if(/^\$PSIBLASTEXE=/){
	print fhRUNNEW "\$PSIBLASTEXE=\"".$params{"-blast"}."\";\n";next;
    }
    if(/^\$NR=/){
	print fhRUNNEW "\$NR=\"".$params{"-nr"}."\";\n";next;
    }
    print fhRUNNEW "$_";

}

close fhRUN;
close fhRUNNEW;
$cmd="mv ".$params{"-home"}."/$fn.new ".$params{"-home"}."/$fn";
`$cmd`;
chmod(0755, $params{"-home"}."/$fn");

}

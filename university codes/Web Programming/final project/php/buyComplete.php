<?php
session_start();
$email = $_SESSION['email'];
$person_id = 0;

$servername = "localhost";
$user = "root";
$password = "";
$dbname = "final_project";

// Create connection
$conn = new mysqli($servername, $user, $password, $dbname);
mysqli_set_charset($conn, "utf8");
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$exist = false;
$sql = "SELECT id FROM person where email=\"" . $email . "\"";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    // output data of each row
    while ($row = $result->fetch_assoc()) {
        $person_id = $row["id"];
    }
}
$sql = "UPDATE buying 
SET 
    status = 1
WHERE
    person_id = " . $person_id . ";";
$result = $conn->query($sql);

$sql = "UPDATE buying 
INNER JOIN good ON good.id = buying.good_id
SET 
    buying.sell_Price = good.price
WHERE
    status=1 and sell_Price is Null;";
$result = $conn->query($sql);


header("location:./basket.php");
?>
